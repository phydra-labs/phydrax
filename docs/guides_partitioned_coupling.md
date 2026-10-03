# Partitioned multiphysics coupling

`phydrax.solver.coupling` composes pure fixed-window subsystem maps without
introducing another mesh, nonlinear-solver, or communication stack. It is intended
for native same-process coupling when subsystem reuse is more important than a
monolithic discrete solve.

Use a specialized or monolithic method instead when participants share algebraic
unknowns, stagewise conservation must hold exactly, or a coupled Jacobian and
preconditioner are available. Existing CFD--DEM, compatible Maxwell--PIC, immersed
boundary, and other transactional physics runtimes remain distinct. Steady problems
whose owners share one monolithic discrete solve across an interface are assembled
by the spatial coupled problems of
[Numerical interoperability](guides_numerical_interoperability.md#spatial-coupled-problems).

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
positive `reference_scale`. The reference scale and the vector-space pairing define
the interface residual norm used for nonlinear scaling and physical certification.
They are not an inventory: a physical amount is a separate `CouplingMeasurement`
functional and is never inferred from the norm, the storage shape, or a cell
measure. Field-valued ports additionally retain their `DiscreteFieldSpace`.
Physically typed ports declare a `CouplingQuantity`, a component `frame` (default
`"scalar"`), a `temporal_kind` of `"instantaneous"` or `"interval_integral"`, and
optionally a `measurement`. Ports without a quantity remain untyped mathematical
endpoints. No exchange broadcasts, reshapes, casts, or selects a mapping.

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

## Physical measurements and ledgers

`CouplingMeasurement` is a prepared linear functional `L` from one port's native
coordinates to an inventory with one entry per declared component:

```python
cpl.CouplingMeasurement(
    functional,                  # AbstractLinearOperator into ArraySpace((m,))
    unit,                        # UnitDefinition of the measure
    representation="density",    # "density", "extensive", or "functional"
    support_id=field.support_id,
    provenance_id="surface-area",
    component_ids=("scalar",),
    normalization="physical",
)
```

`inventory(value)[k]` is the amount of component `k` in the port quantity's unit
times the measurement `unit`. The functional must map into a real floating
`ArraySpace((m,))` and must provide a transposed action; `covector(weights)` pulls
inventory weights back to native coordinates. Construction pulls back every
component covector once on the host and checks the declared representation:

| Representation | Stored coordinates | Construction check |
|---|---|---|
| `"density"` | Densities integrated against a physical measure | `"physical"` normalization, nonnegative weights, positive mass per component |
| `"extensive"` | Amounts that already carry their measure | `"counting"` normalization, dimensionless unit of scale one, disjoint 0/1 covectors |
| `"functional"` | Any declared general linear action, such as basis integrals or flux moments | Finite covectors |

An extensive measurement sums each stored amount exactly once, so coordinates
weighted by a cell measure a second time are refused. `measurement_id`
fingerprints the representation, normalization, unit, components, support,
provenance, source space, and covector content.

Two constructors cover the common cases. `CouplingMeasurement.from_measure(measure,
space, unit, component_ids=...)` integrates entity-major `(entity, component)`
coordinates against one `DiscreteMeasure` through a matrix-free operator with an
explicit transpose. `CouplingMeasurement.extensive(space, support_id,
provenance_id=..., component_ids=...)` sums extensive amounts per component with the
dimensionless unit `ONE`.

A port measurement must act on the port space and belong to the port field
support. Density and extensive measurements must match the field storage:

| `FieldRepresentation` | Storage | Admissible measurements |
|---|---|---|
| `point_value`, `cell_average`, `basis_coefficient` | Density | `"density"` or `"functional"` |
| `cell_integral`, `flux_moment`, `circulation_moment`, `cochain` | Extensive | `"extensive"` or `"functional"` |
| `polynomial_moment`, `modal_coefficient`, `particle_value`, `functional`, `custom` | Declared | `"functional"` only |

Nodal, basis, and flux-moment storage therefore produce inventories without
coercion to cell averages. A measurement with more than one component requires a
non-`"scalar"` component frame; unlike scalar quantities belong to separate ports.
An `"interval_integral"` port requires a quantity and a measurement that is not
probability-normalized, and carries no waveform plan.

Physically typed exchanges are admitted only when:

- both ports are typed, and the received quantity (the conversion's
  `integrated_quantity` for `"integrate"`, otherwise the source quantity) has the
  target's `compatibility_id`;
- measurements at both ends use the same unit dimension and reference system and
  inventory the same component IDs;
- a direct exchange keeps the same frame, field space, and `measurement_id`;
- a `FieldTransfer` exchange declares a `CouplingTransferRequirement` and both
  measurements, and its `frame_action` is `"preserve"` exactly when the frames
  agree;
- a whole-window target uses a conservative requirement;
- a storage change between density and extensive coordinates (the received and
  target quantity dimensions differ, for example J/m² cell averages into J cell
  integrals) goes through a certified conservative transfer; only the certificate
  carries the measure between the two storages.

For a conservative transfer `P`, preparation certifies the declared physical
measurement identity

```text
L_target P = L_source
```

per component. Each target covector is pulled back through the transposed action of
the operator the exchange actually applies (the primal transfer, or its paired
adjoint when `use_adjoint=True`) and compared with the source covector, scaled by the
measurement units. This costs one transposed action per component and never
materializes a dense transfer matrix. A transfer that falsely claims conservation
fails preparation. A source port may feed at most one whole-window exchange, so one
authoritative amount is never spent twice.

`CouplingState.cumulative_exchange_budget` has shape `(row_count, 2)` with one
source-debit/target-credit pair per entry of `budget_row_ids`. There is one row per
exchange, identified by its exchange ID; a whole-window exchange with a
componentized measurement instead contributes one row per component, identified as
`f"{exchange_id}[{component}]"`, so unlike components are never summed. Only
exchanges into `"interval_integral"` targets are budgeted; other rows stay zero. The
debit is `-L_source(amount)` and the credit is `L_target(received)`, each in the
reference units of quantity times measurement (and the graph clock `time_unit` for
integrated rates). Every row is certified at roundoff level, `64·eps` relative to
the larger of the booked amounts and the inventory of a signal at the ports'
declared `reference_scale` (never an absolute floor of one SI unit), together with
local agreement between the received and mapped values relative to the target's
`reference_scale`; a violation is a `CERTIFICATION_FAILURE`. Epoch transitions map
budgets by row ID and refuse to change a retained exchange's temporal conversion,
clock unit, quantity compatibility, temporal kind, frame, or measurement unit,
representation, and components.

## Explicit temporal conversions

`CouplingExchange(..., temporal=CouplingTemporalConversion(kind, ...))` declares how
a source signal becomes the target signal. Preparation derives the one admissible
kind from the two ports and refuses any mismatch: temporal meaning is never
inferred.

| Source port | Target port | Required `temporal` |
|---|---|---|
| Instantaneous endpoint | Instantaneous endpoint | `None` |
| Waveform | Instantaneous endpoint | `CouplingTemporalConversion("sample-end")` |
| Instantaneous endpoint | Waveform | `CouplingTemporalConversion("hold")` |
| Waveform | Waveform | `CouplingTemporalConversion("interpolate", transfer=BarycentricCouplingTemporalTransfer(k))` |
| Interval integral | Interval integral | `CouplingTemporalConversion("window-integral")` |
| Waveform rate | Interval integral | `CouplingTemporalConversion("integrate", integrated_quantity=...)` |

- `"sample-end"` supplies the source waveform's value at the window end.
- `"hold"` holds an endpoint value constant across the target waveform grid.
- `"interpolate"` reconstructs the source waveform on the target grid with the
  declared temporal transfer, which never extrapolates.
- `"window-integral"` passes an authoritative whole-window amount without temporal
  reinterpretation.
- `"integrate"` multiplies the window size by the integral of the source waveform's
  own reconstruction. The reconstruction is one polynomial of the plan's
  `polynomial_degree` per interval, so a `⌈(degree + 1) / 2⌉`-point Gauss rule
  integrates it exactly, independent of the residual `metric_order`.
  `integrated_quantity` must have the source rate dimension times the clock
  `time_unit` and the target's compatibility. The clock unit is declared once as
  `CouplingGraph(..., time_unit=...)` (or on the `PartitionedCouplingDeclaration`);
  a graph with an `"integrate"` exchange refuses to be built without it, and the
  unit is part of the graph identity.

Instantaneous endpoints and interval integrals cannot be exchanged in either
direction: an endpoint value is never multiplied by the window and reported as an
exact integral. A whole-window amount has no waveform history and cannot be held
across a waveform.

## Pure participant callback

`CallableCouplingSubsystem` adapts a callback with signature

```python
advance(window, start_state, inputs, args) -> CouplingSubsystemResult
```

`inputs` and `outputs` are tuples aligned with the declared ports. The result reports
one candidate state, scalar success/status/residual/iteration/work evidence, and
optional fixed-structure native `evidence`: an array PyTree the participant
publishes for accepted and refused evaluations alike.

Preparation uses shape evaluation to prove that the callback preserves participant
state structure and returns every declared output space. Native participants must be
JIT-capable and fixed-topology; host participants run only on the
[explicit host route](#host-participants). Participants in implicit cycles must additionally
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
is re-evaluated once with the same sweep before acceptance, and the physical exchange
residual of that final evaluation decides certification. Under a Gauss--Seidel sweep a
consumer ordered after its source therefore receives exactly the whole-window amount
the source spent in that sweep, so the ledger balances to roundoff rather than to the
interface tolerance. This path intentionally exposes no implicit derivative contract.

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
retains physical residuals, thresholds, participant statuses, participant work and
iterations, exact participant evaluation counts, transfer applications, and coupling
iterations. `counts_complete` is true when the work of every executed evaluation is
counted and every participant reports complete counts: explicit sweeps evaluate each
participant once, and the fixed-point path declares each interface evaluation's
participant work to `FixedPointIteration`, which sums it over every iterate it
executes, including safeguarded Anderson re-evaluations. Participant work then equals
that sum plus the final certification evaluation. General-root methods do not report
per-evaluation work, so their windows count only the final evaluation and report
`counts_complete=False`; that work is never inferred.

`participant_evidence` retains each participant's native `evidence` from the
evaluation defining the candidate, ordered by `subsystem_ids`, whether the window
commits or rolls back. Participants keep their own structures; nothing is summed
or maximized across participants.

## Fixed-window rollout

`CouplingProblem` requires an exact integer number of fixed coupling windows and
explicit initial values for every exchange. `CouplingRolloutPlan` provides final,
checkpoint, or trajectory retention and reuses `FixedStepReplayPolicy` for deterministic
reverse recomputation. Its `evidence_retention` bounds retained participant
evidence: `"terminal"` (default) keeps the last committed window and the refusing
window as separate records, `"steps"` adds one record per window with a committed
mask, and `"none"` keeps nothing.

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
stencils and never extrapolates. It is declared on the exchange through
`CouplingTemporalConversion("interpolate", transfer=...)`; ports carry no temporal
transfer. The physical waveform norm integrates the vector-space pairing of the port
plan's own piecewise polynomial reconstruction, of degree `polynomial_degree`, with
its exact declared Gauss order `metric_order`; degree-one all-active data recovers
the fixed linear behavior. Adaptive defects activate a deterministic candidate and
restart every participant from the unchanged window checkpoint. Exhausted capacity
returns `CouplingWaveformCapacityRequest`; growth occurs only in a host-prepared
epoch.

`AdaptiveCouplingWindowPolicy` accepts only physically certified windows with
reliable participant `CouplingWindowErrorEstimate` ratios at most one. Its bounded
PI controller retries only declared statuses, saturates at explicit minimum and
maximum sizes, and reaches final time exactly through the canonical fixed-segment
runner. Every rejected attempt restarts the same checkpoint, so
`rollout_adaptive_coupling` refuses any participant without deterministic replay. A
window that succeeds but lacks a reliable, finite local error estimate from every
participant cannot be accepted or controlled; the rollout stops at the retained
checkpoint with terminal status `UNRELIABLE_ERROR_ESTIMATE`.

`AdaptiveCouplingSolution` reports the work of every executed attempt, not only
accepted work: `attempted_windows`, `rejected_attempts`, and `participant_work`,
`participant_evaluations`, and `coupling_iterations` summed over all attempts,
including rejected ones. `maximum_rejected_error_ratio` retains the largest reliable
rejected local error ratio. Its static `counts_complete` follows the window rule
above: false for general-root implicit policies or when any participant reports
incomplete counts.

`PreparedCouplingEpoch` and `CouplingEpochTransitionPlan` apply topology or
source-owned remesh requests only after an accepted window. Retained values need
an explicit transfer; added and removed participants need an initializer or
finalizer. Missing routes, coverage, or conservation retain the complete old
graph/state epoch. Frozen event replay is required across node, window, and
topology decisions.

When a topology change also touches owners outside the coupling graph
(discretizations, factorizations, observations, interface routes), the whole
composition is rebound in one `phx.lifecycle` transaction instead; see the
adaptive workflow below.

## Native method participants

Existing native owners bind as participants without a reimplemented step. Each
derives `AbstractMethodCouplingParticipant`, whose `initial_outputs(state, args)`
returns outputs consistent with an accepted checkpoint: instantaneous ports observe
it, waveform ports hold that observation across their grid, and whole-window ports
carry zero amount.

The checkpoint is `MethodParticipantState(native, model_state, key_data,
accepted_windows, native_steps)`: the owner's complete native continuation, its
explicit model state, the data of a carried typed PRNG key, the committed window
count, and the owner's native step index at which the continuation resumes. Every
window evaluation, implicit interface iterate, and adaptive retry starts from this
checkpoint; the candidate advances each field once per evaluation and is committed
only on acceptance. A rejected candidate therefore never leaks history, model
state, or randomness, and replay reproduces it. `randomness` is declared as
`MethodParticipantRandomness`:

- `"none"`: the owner is deterministic;
- `"carried-key"`: the checkpoint carries a key of implementation `key_impl`; each
  window splits it into a window key and the next checkpoint key;
- `"external"`: randomness outside the checkpoint. The participant reports no
  deterministic replay, so implicit cycles and `rollout_adaptive_coupling` refuse
  it.

`FixedStepCouplingParticipant(method, bind, observe, subsystem_id=..., substeps=...)`
binds an `AbstractFixedStepMethod` or `ConservationIMEXMethod`. Each window takes
`substeps` uniform native steps with the owner's own step and evidence; after a
failed step no further owner work is performed. Native step `k` of a window receives
the global step index `native_steps + k`, so index-dependent owners (such as
production schedules that check their accepted step) continue across windows;
`initial_state(native, step_index=...)` declares the index at which `native`
resumes (default 0). `bind(sub_window, views,
model_state, key, args)` returns `MethodWindowBinding(method_args, model_state)`
from the declared per-substep input views:

| Input port | Per-substep view |
|---|---|
| Instantaneous endpoint | The window value, frozen |
| Waveform | `(start, end)` samples at the substep's native nodes |
| Interval integral | The whole-window amount divided by `substeps` (uniform rate) |

Waveform ports sample exactly the native nodes `linspace(0, 1, substeps + 1)` with
no adaptation; an interpolating exchange changes grids. `observe(native, args)`
returns every instantaneous output in port order, observed at every node for
waveform ports. `amounts(start_native, end_native, args)` returns every whole-window
output and is required exactly when such ports exist. The owner must itself have
spent those amounts, for example through a native flux accumulator; the participant
does not compute a debit. An optional `estimate_error` supplies the adaptive window
error.

A fixed-step participant publishes `FixedStepParticipantEvidence(executed,
successful, method)`: per-substep masks of the native steps that ran and that the
owner accepted (the first executed unaccepted substep is the refusal), and the
owner's evidence over the window. Only the owner reduces its substep evidence,
through `AbstractFixedStepMethod.reduce_evidence`; an owner declaring no
reduction keeps the bounded per-substep stack, as does conservative IMEX stage
evidence. A refused window keeps this evidence while its checkpoint is unchanged.

`DAECouplingParticipant(prepared, bind, observe, subsystem_id=...)` binds an
adaptive, event-free `PreparedDAESolve`. Its prepared time grid is a template whose
normalized save times are rebound to each window. The checkpoint carries the array
part of the owner's `DAEContinuation`, including the retained nonlinear solve, as
`DAEParticipantNative`; the first window runs the owner's initialization. Fixed-grid
and event-driven DAE solves cannot resume their history and are refused. Inputs are
endpoint or whole-window values passed through `bind`; waveform inputs are refused.
Participant work counts every accepted and rejected DAE attempt.

`SteadyResponseCouplingParticipant(respond, subsystem_id=..., input_ports=...,
output_ports=..., initial_response=...)` binds a steady field-response owner.
`respond(inputs, model_state, key, args)` returns a `SteadyResponse` and never
receives the window. A steady response is not a transient solver: it has no
whole-window ports, its ports are all endpoints or all share one waveform plan by
`plan_id` (responding at every active node), and its temporal error is zero rather
than an estimate of transient physics. Its checkpoint `native` is the accepted
end-of-window response, starting from `initial_response`, so a restart or
re-preparation from an accepted checkpoint observes that response.

## Declaring and lowering a partitioned problem

`PartitionedCouplingDeclaration(participants, exchanges, policy, differentiation=...,
resources=..., time_unit=...)` declares participants, physical exchanges, one
temporal policy, and the coupling clock unit once. It owns no window loop.
`lower_partitioned_coupling` binds one initial checkpoint per participant and
returns the canonical `CouplingProblem`:

```python
declaration = cpl.PartitionedCouplingDeclaration(
    (ocean, atmosphere),
    exchanges,
    cpl.ExplicitCouplingPolicy(cpl.CouplingSweep("jacobi")),
)
problem = cpl.lower_partitioned_coupling(
    declaration,
    {
        "ocean": ocean.initial_state(ocean_native),
        "atmosphere": atmosphere.initial_state(atmosphere_native),
    },
    t0=0.0,
    t1=1.0,
    window_size=0.1,
)
solution = cpl.solve_coupling(problem)
```

Every exchange whose source is a native method participant starts from that
source's checkpoint observation mapped through the declared spatial transfer, unit
conversion, and temporal conversion; whole-window amounts start at zero, and these
values cannot be overridden. Exchanges from other participants require explicit
`exchange_values`. Preparation proves the exchange routes before any initial value is
mapped, and execution uses the existing partitioned runtime.

## Mixed-method conjugate heat workflow

`examples/mixed_method_time_coupling.py` couples two native owners across a
nonmatching interface, each with its own integrator and step count per window:

- a P1 finite-element solid on `[0, 1]²`. Its backward Euler step is an
  `AbstractFixedStepMethod` that solves `(ρc M + Δt k K) Tⁿ⁺¹ = ρc M Tⁿ + Sᵀℓ`.
  The operator is compiled once from `MassAction` and `DiffusionAction` and factored
  once by `phx.linalg.prepare`, and every step reuses that factorization. A step
  with any other size reports failure;
- a cell-centered finite-volume fluid on `[1, 2] × [0, 1]` advanced by
  `SSPRK33FixedStepMethod`. Its native state holds the cell temperatures and the
  accumulated heat that has left each interface face, so the whole-window amount is
  owner-spent: `amounts(start, end, args)` returns the accumulator increment.

Four solid edges meet three fluid faces on `x = 1`. `I` holds the exact face averages
of the solid interface hat functions:

| Port | Storage | Temporal meaning | Measurement |
|---|---|---|---|
| `solid-fe/interface-temperature` | `basis_coefficient` (P1 trace) | waveform on the solid step nodes | none |
| `fluid-fv/interface-temperature` | `cell_average` (face averages) | waveform on the fluid step nodes | none |
| `fluid-fv/face-heat` | `flux_moment` | `interval_integral` | `CouplingMeasurement.extensive` |
| `solid-fe/interface-loads` | `functional` (basis loads) | `interval_integral` | counting sum of the loads (`representation="functional"`) |

The two exchanges and the implicit window policy are declared once and lowered:

```python
exchanges = (
    cpl.CouplingExchange(
        "interface-temperature",
        "solid-fe/interface-temperature",
        "fluid-fv/interface-temperature",
        transfer=face_averages,  # FieldTransfer with action I
        requirement=cpl.CouplingTransferRequirement(constant_preserving=True),
        temporal=cpl.CouplingTemporalConversion(
            "interpolate", transfer=cpl.BarycentricCouplingTemporalTransfer(1)
        ),
    ),
    cpl.CouplingExchange(
        "interface-heat",
        "fluid-fv/face-heat",
        "solid-fe/interface-loads",
        transfer=basis_loads,  # FieldTransfer with action P = Iᵀ
        requirement=cpl.CouplingTransferRequirement(conservative=True),
        temporal=cpl.CouplingTemporalConversion("window-integral"),
    ),
)
policy = cpl.ImplicitCouplingPolicy(
    phx.nonlinear.FixedPointIteration(),
    phx.nonlinear.NonlinearTermination(
        absolute_residual=1e-11,
        relative_residual=0.0,
        absolute_step=0.0,
        relative_step=1e-14,
        maximum_steps=60,
    ),
    (
        cpl.CouplingTolerance("fluid-fv/interface-temperature", absolute=1e-10),
        cpl.CouplingTolerance("solid-fe/interface-loads", absolute=1e-10),
    ),
    fixed_point_sweep=cpl.CouplingSweep(
        "gauss-seidel", subsystem_order=("fluid-fv", "solid-fe")
    ),
)
problem = cpl.lower_partitioned_coupling(
    cpl.PartitionedCouplingDeclaration((solid, fluid), exchanges, policy),
    {
        "solid-fe": solid.initial_state(solid_temperatures),
        "fluid-fv": fluid.initial_state(fluid_native, model_state=fluid_model),
    },
    t0=0.0,
    t1=0.4,
    window_size=0.05,
)
solution = cpl.solve_coupling(problem)
```

Preparation certifies `L_solid P = L_fluid` through the transposed action. The rows
of `I` sum to one, so the total of the nodal loads equals the total face heat. The
Gauss--Seidel order runs the fluid first, so the solid consumes exactly the heat the
fluid spent in the same sweep, and the iteration closes only the temperature waveform.
Fixed-point termination sets explicit step thresholds: the default
`relative_step=1e-10` would report stagnation while the contracting interface is
still above `absolute_residual`.

The example compares both owner pairings with `scipy.linalg.expm` of the assembled
coupled semi-discrete system (host P1 matrices and the two-point fluid stencil) at
`t = 0.4`. It checks first-order convergence in the window size, a balanced
`interface-heat` ledger row, total energy conserved to roundoff, and transactional
rollback. The observed maximum nodal/cell errors are:

| Local integrators per window | `Δw = 0.1` | `0.05` | `0.025` | Observed rates |
|---|---|---|---|---|
| backward Euler x2, SSPRK(3,3) x3 | `1.03e-2` | `4.92e-3` | `2.39e-3` | 1.06, 1.04 |
| backward Euler x4, SSPRK(3,3) x1 | `5.72e-3` | `2.60e-3` | `1.23e-3` | 1.14, 1.08 |

The rollback check gives the fluid a carried key (a lognormal conductance draw per
step) and model state (conductance and applied steps). A plan with `maximum_steps=2`
ends `WORK_EXHAUSTED`. Its accepted state is the checkpoint, while its candidate has
advanced the key once and the model state by three steps. Replaying it is bitwise
identical. A converging plan commits the window once, whatever the iterate count, and
its diagnostics count the work of every iterate with `counts_complete=True`.
`tests/integration/test_mixed_method_time_coupling.py` asserts the same contracts.

## Adaptive interface rebind

`examples/adaptive_fe_fv_rebind.py` refines one side of the mixed-method
conjugate-heat coupling at the accepted boundary `t = 0.2` and continues the
accepted windows on the new topology. The coupling owner contributes three hooks
to the lifecycle transaction (`phx.lifecycle.CompositionRebind`):

- `coupling_composition_entries(epoch, state, *, native_dependencies,
  discretization_independent, epoch_dependencies)` splits an accepted
  `CouplingState` into `<participant>/native` (structure: the participant's
  declared `discretization_bundle_id`), `<participant>/model-state`,
  `<participant>/rng`, `<participant>/windows`, `exchange/<exchange>`,
  `coupling/budget` (meaning: the physical budget contract of every exchange),
  `coupling/clock`, and `coupling/epoch`, which binds all of them and the owner
  artifacts it was prepared from. Model state and carried keys bind their
  participant's native structure unless the participant is declared
  discretization-independent, so a grid change without an explicit transport is
  refused rather than carried silently;
- `coupling_exchange_transport(source, target_epoch, target_entries, exchange_id)`
  re-derives a changed exchange boundary value through the target epoch's declared
  route; the consumed content of past windows stays in the retained ledger;
- `coupling_state_from_composition(composition)` assembles the published epoch and
  accepted state, with the clock and cumulative budgets unchanged.

A staged solid refinement lowers the refined participant and the retained fluid
participant from the transported checkpoints at `t = 0.2`
(`lower_partitioned_coupling`), reprepares the interface route and prepared
epoch, and publishes through `commit_composition_rebind` only for an accepted
window. The fluid state object is retained unchanged, heat content is unchanged
by the rebind to roundoff, and the continued run converges to the rebound
semi-discrete reference at first order.
`tests/integration/test_adaptive_fe_fv_rebind.py` and
`tests/integration/test_coupled_restart.py` assert the rebind, refusal, and
checkpoint-restart contracts.

## Host participants

A host participant (`AbstractHostCouplingParticipant`, for example
`phydrax.interchange.fmi.FMICouplingParticipant`) executes outside JAX with visible
process I/O. It declares the same ports, quantities, and measurements as a native
participant and can sit in the same `PartitionedCouplingDeclaration`, but
`prepare_coupling`, `lower_partitioned_coupling`, `advance_coupling_window`, and
`refresh_coupling` refuse it. `prepare_host_coupling(declaration, native_states,
t0=..., window_size=...)` is the only boundary that executes it:

- Preparation applies the canonical route, physical-exchange, temporal-conversion,
  conservative-certificate, and policy validation. It requires at least one native
  and one host participant; an all-native graph belongs to the native runtime.
  Host participants start from their live communication point, which must be `t0`,
  and their initial outputs derive initial exchange values like method participants.
  Every host participant runs on the declaration's coupling clock: the
  declaration must set `time_unit`, and each host participant's `time_unit` (for
  FMI, the binding's checked independent-variable unit) must equal it.
- `advance_host_coupling_window` and `solve_host_coupling` execute the declared
  sweep eagerly: native participants run their window maps compiled once, and host
  participants perform one host window each. Window `k` spans
  `[t0 + k·h, t0 + (k + 1)·h]` on the prepared grid rather than an accumulated
  clock, and the last window of `solve_host_coupling` ends exactly at `t1`, so a
  host process is never stepped past its declared stop time. Window status,
  interface certification, ledger rows, diagnostics, and the accepted/candidate
  state are the native window contract; each host participant contributes a
  `HostParticipantState` marker (accepted time and windows) to the `CouplingState`.
- Lifecycle: before a window every host process must sit at the accepted
  communication point. With `rollback="restore"` (a real save/restore of the complete
  host state, such as FMI `canGetAndSetFMUstate`), each such participant is
  captured individually, a rejected window restores it, and an accepted one releases
  the checkpoint. Without restore only an explicit, non-retrying policy is admitted.
  A rejected window is `commit="rejected"` when every participant without restore
  stayed at the accepted point, and `commit="unrecoverable"` otherwise; restorable
  participants are restored in both cases, and the next window after an
  unrecoverable one refuses to start. When a host call raises, every captured
  participant is restored and released independently; the original error
  propagates, and any restore or release failure (for example of a session the
  failing call closed) is attached to it as a note.
- Implicit host windows require restore for every host participant and a damped
  `FixedPointIteration` without acceleration. Every iterate after the first restores
  the checkpoint before re-executing; the last evaluation is the certified candidate,
  and termination classes follow the native damped fixed-point method. General-root
  methods and Anderson acceleration trace the participant map and are refused.
- Host coupling forms no derivatives. A differentiation request is refused with each
  host participant's `derivative_support`; a forward co-simulation does not acquire
  an adjoint by wrapping it, and a staged `ExternalAdjointAction` is never chained
  through host windows. JAX transformations of host windows are refused.

`HostCouplingWindowResult` carries the canonical `CouplingWindowResult`, the commit
decision, host evaluation and restore counts, and the host evidence IDs (the FMI
session artifact). `examples/fmi_host_coupling.py` couples a compiled FMU zone to a
native node on both routes.

## Meshfree bulk–surface windows

`MeshfreeComponent` publishes native point-cloud or intrinsic-surface
reconstruction and capacity without pretending to have a facet trace.
`SurfaceExchangeLaw` pairs a complete constant-reproducing bulk query with its
exact coordinate transpose, preserving bulk loss and surface gain.
`MeshfreeBulkSurfaceMethod` composes the existing conservative Langmuir solver
with a positive deposition route and binds `FixedStepCouplingParticipant`.
Each step publishes `MeshfreeBulkSurfaceEvidence`: native film and nonlinear
statuses, nonlinear work and residual, Langmuir coverage, the amount actually
transferred (spent only by accepted steps) beside the raw candidate transfer, the
closed-system amount defect, complete query coverage, and whether the surface
metric was admitted exactly or relaxed. Its declared window reduction sums spent
transfers and work, keeps extremal coverage and residuals, and reports the
refusing substep's status, so window and rollout evidence balance the amounts
the participant actually moved.

Host window boundaries relocate moving surface queries. Their sites remain
frozen within the window; displacement, lag, and conservation are reported.
No within-window geometry derivative is implied. A topology epoch separately
transfers extensive histories atomically through native lifecycle owners.
See [Meshfree solvers](guides_meshfree.md#bulk-surface-exchange) and
`examples/meshfree_bulk_surface_exchange.py`.

The same `SurfaceExchangeLaw` also joins a monolithic transient: with a
finite-element bulk, a meshfree surface, and `LangmuirAdsorptionFlux`, it lowers
through `prepare_coupled_transient` onto the native index-one DAE, whose BDF
stages Newton-solve the nonlinear exchange. `CoupledTransientSolution` admits the
native discrete implicit derivatives only at an accepted solution
(`derivative_valid`); derivatives through a refused trajectory are NaN.

## Mixed-method meshfree interfaces

A point cloud has no facets of its own. `PointBoundaryCharts` takes them from a
geometry boundary atlas whose Gauss–Lobatto sites must coincide with cloud
points, and `MeshfreeTraceComponent` publishes the cloud's coupling boundaries
through that authority. Collocated coupling rows are homogeneous Neumann conormal
rows scaled by the charts' lumped measure; the owner's residual reaction is the
canonical conormal flux, and the value trace is the charts' nodal interpolant.
Scalar dissipative rows already contain the integrated weak balance and are not
rescaled. Their natural boundary nodes retain volume capacity/reaction weights;
only Dirichlet rows exclude those terms. Floating weak-owner modes reach the
coupled gauge consumer only after both actual reduced operator and transpose
actions verify the owner-declared candidates.
Rows, normals, and measures that disagree with the charts are refused, as are a
pointwise flux and a trace-inverse certificate, so a Nitsche side on a cloud must
carry zero flux weight. The cloud then couples through the ordinary laws:
`ScalarTransmissionLaw` with a mortar in either orientation or a one-sided
Nitsche against finite elements, and `ConservativeFluxLaw` with an
`InterfaceConductance` against a `FiniteVolumeComponent`. The owner must itself
be admitted by its spectral stability assessment. For traction interfaces,
prepare `PointGhostLayerPlan(boundary).prepare(cloud)` and pass it as `ghosts`
to `PointBlockSystemPlan`: this preserves the cloud PDE at every interface
point and imposes traction on the corresponding ghost equation. Coupling
publishes each cloud field and its `<field>-ghost` unknowns, exchanges the
traction and PDE rows, and scales traction by boundary measure times ghost
offset. The original extended equations, including cross-component stress
terms, remain the coupled residual.

In mixed Stokes owners, component-specific traction anchors pressure only where
that component owns a nonzero normal contribution. Tangential-only traction does
not remove the pressure gauge.

`VectorTransmissionLaw` couples vector fields (velocity, displacement rate)
component by component; `VectorTransmissionCertificate.resultants` reports the
integrated interface force and each side's power as a dual pairing of the
traction covector with that side's trace, never a sum of vector coefficients.

Pass the complete `dict(solution.fields)` to `resultants`, including ghost
fields. The FSI example computes solid elastic power with the same prepared
ghost-extended derivative family and the complete displacement-rate state;
cloud-only derivatives do not represent the solved traction discretization.
See `examples/meshfree_mixed_method_coupling.py` and
`examples/meshfree_fluid_structure.py`.

## Statuses

| Status | Meaning |
|---|---|
| `SUCCESS` | Explicit sweep completed or implicit root physically certified. |
| `PARTICIPANT_FAILURE` | At least one participant rejected its window evaluation. |
| `NONFINITE_EVALUATION` | Participant, exchange, or residual data became nonfinite. |
| `NONLINEAR_FAILURE` | The selected implicit nonlinear method failed. |
| `WORK_EXHAUSTED` | Nonlinear step, evaluation, or inner-linear budget was exhausted. |
| `CERTIFICATION_FAILURE` | Final physical interface residual exceeded a port tolerance, a whole-window ledger row did not balance, or an adaptive window still exceeded its error tolerance at the minimum size. |
| `UNRELIABLE_ERROR_ESTIMATE` | Adaptive acceptance lacked a reliable finite local error estimate for a successful window. |

There is no success alias for an exhausted or uncertified implicit solve.

## Unsupported boundaries

The native substrate does not provide process communication, MPI/socket routing,
external mutable participants, XML configuration, automatic mapping selection,
hidden iterate clipping, or fallback solvers. Topology changes are accepted-boundary
host transitions over explicitly prepared graphs and source-owned transfers, never
an in-trace graph mutation. External mutable participants execute only on the
explicit host route above and are never a native JIT or differentiation path.

See `examples/partitioned_coupled_oscillators.py` for an implicit differentiable
example, `examples/mixed_method_time_coupling.py` for the mixed-method workflow above,
and `tools/partitioned_coupling_qualification.py` for executed convergence,
acceleration, residual, and derivative evidence.
