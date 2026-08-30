# Differential-algebraic equation integration

`phydrax.solver` integrates regular finite-dimensional implicit systems of the form

\[
F(t, y, \dot y, a) = 0
\]

on a declared `TimeGrid`. The implementation is native Phydrax: consistency and every
implicit stage use the prepared nonlinear and linear-solve lifecycles from
`phydrax.nonlinear` and `phydrax.linalg`. It does not route through Optimistix or
Diffrax.

This path covers regular index-one problems on either a declared fixed grid or an
adaptive accepted-step grid. It does not infer a differentiation index, reduce a
higher-index system, or handle events. Those capabilities require separate structural
and trajectory contracts rather than hidden heuristics in the integrator.

## System and structure

`DifferentialAlgebraicSystem` owns only the state-shaped residual and its invariant
metadata. Initialization and integration policy belong to the problem and solver.
`DAEStructure` declares differential/algebraic roles for variables and equations along
one component axis. For an unstructured state, use one role with
`component_axis=None`.

The residual, state, and state rate must all have `state_shape`. `state_scale`,
`state_rate_scale`, and `residual_scale` are independent positive, finite, real arrays
broadcast to that shape:

- state and state-rate scales define the coordinate weights used by consistency and
  BDF nonlinear solves;
- residual scale defines the equation weights used by convergence checks;
- none of the scales changes the raw value returned by `system(...)`.

`DifferentialAlgebraicSystem` also accepts an optional `InputLayout`. An autonomous
residual keeps the signature `residual(time, state, state_rate, args)`. An input-aware
residual uses `residual(time, state, state_rate, inputs, args)` and must be evaluated
with keyword-only `inputs`. Missing, extra, or incorrectly shaped inputs fail before
the residual runs. `from_mass_matrix` supports the same input-aware vector-field form;
the mass matrix itself remains independent of the input in this contract.

`DifferentialAlgebraicProblem` binds an `AbstractInputPolicy` whenever the system is
input-aware. Autonomous problems reject a policy; input-aware problems require one with
the exact `InputLayout` identity. The policy is evaluated at every consistency candidate,
implicit stage, residual certification, fixed/adaptive replay point, and continuation
boundary, so state-dependent policy derivatives enter native Jacobians. Continuations
retain and verify the policy ID. `HeldInputPolicy` provides state-independent,
piecewise-constant interval values with an explicit internal-node convention.

Nontrivial state geometry is rejected by the native BDF backend. The current BDF
formula combines ambient Euclidean states and therefore cannot honestly preserve a
manifold-valued state.

```python
import jax
import jax.numpy as jnp
import phydrax as phx


def residual(time, state, state_rate, decay):
    del time
    differential = state_rate[0] + decay * state[0]
    constraint = state[1] - state[0] ** 2
    return jnp.stack((differential, constraint))


system = phx.dynamics.DifferentialAlgebraicSystem(
    residual,
    state_shape=(2,),
    structure=phx.dynamics.DAEStructure(
        ("differential", "algebraic"),
    ),
    state_scale=jnp.asarray((1.0, 1.0)),
    state_rate_scale=jnp.asarray((1.0, 1.0)),
    residual_scale=jnp.asarray((1.0, 1.0)),
    system_id="decay-with-constraint",
)
```

`DifferentialAlgebraicSystem.from_mass_matrix` is a convenience constructor for
`M(t, y, a) @ ydot - f(t, y, a) = 0`. The caller still supplies the structural roles;
a singular mass matrix alone does not prove index-one regularity.

## Consistent initialization

`DifferentialAlgebraicProblem` stores guesses for both `y(0)` and `ydot(0)` and an
explicit `DAEInitializationSpec`. The default is `index_one()`:

- fix differential state components;
- solve algebraic state components;
- solve differential rate components;
- fix algebraic rate components while marking those rates semantically unavailable.

The other modes are explicit:

- `fixed_rate_state()` fixes the entire supplied state and solves the entire rate;
- `check_only()` changes nothing and only checks the supplied pair;
- `from_masks(fixed_state, fixed_rate)` defines custom fixed/free scalar masks.

A custom contract must expose exactly one free state-or-rate scalar per residual
scalar. Masks operate on flattened scalar coordinates after structural broadcasting;
they do not silently choose a least-squares initialization.

```python
problem = phx.solver.DifferentialAlgebraicProblem(
    system,
    jnp.asarray((1.0, 0.0)),
    initial_state_rate=jnp.zeros(2),
    args=jnp.asarray(0.5),
    initialization=phx.solver.DAEInitializationSpec.index_one(),
    problem_id="decay-with-constraint-ivp",
)

initialization = phx.solver.initialize_dae(problem, 0.0)
assert bool(initialization.valid)
```

`DAEInitializationResult` returns the consistent state and rate, their corrections,
fixed masks, componentwise rate validity, scaled residual threshold and norms, native
nonlinear status and diagnostics, and a stable initialization ID. A check-only result
has no nonlinear solve. Failed consistency is reported as evidence; the solver does
not replace a failed pair with the original guess.

## Fixed and adaptive implicit lifecycle

`DAESolvePolicy.method` accepts `BDFMethod(maximum_order=1..5)` or a fixed-grid
stiffly accurate endpoint `ThetaMethod`. Native `NewtonKrylov` or
`NewtonTrustRegion` methods own initialization and stages alongside termination,
temporal reuse, regularity, replay, adjacent-step-ratio, and failure policies.
Supplying `DAEAdaptivePolicy` enables accepted-step BDF control; endpoint theta remains
fixed-grid.

BDF startup increases order only after sufficient accepted history exists. Rejections
or unsafe adjacent-step ratios lower the realized order. Every accepted state is
followed by an independent residual certification.

Planning, preparation, and execution are separate:

1. `plan_dae` validates the grid, method, state contract, and static identities.
2. `prepare_dae` prepares reusable symbolic nonlinear/linear templates for the
   consistency problem and every implicit stage.
3. `solve_dae(prepared, ...)` solves consistency for the runtime state, rate, and
   model arguments, then refreshes numeric stage and Jacobian data while preserving
   both template identities.

```python
grid = phx.dynamics.TimeGrid(
    jnp.linspace(0.0, 1.0, 21),
    time_id="training-grid",
)
policy = phx.solver.DAESolvePolicy(
    method=phx.solver.BDFMethod(2),
    nonlinear_method=phx.nonlinear.NewtonKrylov(),
    nonlinear_termination=phx.nonlinear.NonlinearTermination(
        absolute_residual=1e-9,
        relative_residual=0.0,
        maximum_steps=20,
    ),
)
prepared = phx.solver.prepare_dae(problem, grid, policy=policy)
solution = phx.solver.solve_dae(prepared)
```

The fixed-grid prepared path supports JIT, JVP, VJP, and `vmap`. Each successful
stage state has implicit derivatives obtained from that stage's residual Jacobian;
nonlinear iteration history and diagnostics are nondifferentiable evidence.
Dependencies on previous accepted states are chained through the outer fixed-length
scan. Grid times are stop-gradient values: derivatives are for the declared discrete
fixed-grid map.

```python
def terminal_state(decay):
    solved = phx.solver.solve_dae(prepared, args=decay)
    return solved.states[-1, 0]

terminal_gradient = jax.jit(jax.grad(terminal_state))(jnp.asarray(0.5))
```

## Adaptive integration and accepted-step evidence

`TimeGrid` remains the requested output schedule for an adaptive solve. The controller
may take multiple internal steps between adjacent requested times and lands exactly on
every save boundary. Error weights apply only to differential variables; algebraic
equations are checked independently against `constraint_tolerance`. Residual,
nonlinear, linear, nonfinite, stale-Jacobian, regularity, and local-error failures have
distinct attempt statuses. Accepted-step, attempt, and consecutive-rejection
capacities are hard JAX-static bounds.

```python
adaptive_policy = phx.solver.DAESolvePolicy(
    method=phx.solver.BDFMethod(5),
    nonlinear_method=phx.nonlinear.NewtonKrylov(),
    adaptive=phx.solver.DAEAdaptivePolicy(
        relative_tolerance=1e-5,
        absolute_tolerance=1e-8,
        maximum_accepted_steps=1024,
        maximum_attempts=2048,
    ),
    temporal_reuse=phx.solver.DAETemporalReusePolicy(
        maximum_jacobian_age=4,
        maximum_alpha_ratio=2.0,
        refresh_after_iterations=4,
    ),
    replay=phx.solver.DAEReplayPolicy("chunked", chunk_size=32),
    regularity=phx.solver.DAERegularityPolicy(
        "periodic",
        interval=8,
        failure="status",
    ),
)
adaptive_prepared = phx.solver.prepare_dae(
    problem,
    grid,
    policy=adaptive_policy,
)
adaptive_solution = phx.solver.solve_dae(adaptive_prepared)
```

`DAETemporalReusePolicy` retains numerical Jacobian preparation and, when configured
on the Newton linear policy, GCRO-DR recycling state across related stages. Reuse is
permitted only within the configured Jacobian age, BDF coefficient ratio, and previous
Newton-work bounds. A failed reused stage is retried after a mandatory numerical
refresh. Eisenstat--Walker forcing and remaining aggregate work limits are passed as
dynamic controls into the prepared native Krylov solve; they do not change its static
loop or basis shape.

`DAEStepHistory` records accepted times, step sizes, BDF orders, local error ratios,
source attempt indices, and requested-time mappings. `DAEAttemptHistory` records every
attempt and its nonlinear and linear work. Both histories have fixed capacity plus an
explicit valid mask and count; padded entries are not observations.

## Segmented continuation

Every successful adaptive result carries a `DAEContinuation` boundary object with the
exact accepted BDF history, controller state, and retained nonlinear preparation.
Pass it to a prepared solve whose first requested time equals that boundary:

```text
first = phx.solver.solve_dae(first_prepared)
second = phx.solver.solve_dae(
    second_prepared,
    continuation=first.continuation,
)
```

Continuation requires matching problem, system, state, dtype, integration,
initialization, nonlinear-method, and stage-linear-plan identities. It cannot be
combined with `initial_state` or `initial_state_rate`. These checks prevent a restart
from silently applying BDF history or a recycled numerical operator to a different
model.

## Frozen accepted-grid derivatives and replay memory

Adaptive JVPs and VJPs first run the controller, then stop gradients through accepted
times, step sizes, orders, and save mappings. The accepted stages are replayed with
implicit root derivatives. The resulting derivative is the frozen accepted-grid
discrete derivative; it intentionally excludes derivatives of accept/reject decisions
and the step-size controller. Tightening the primal tolerance is the convergence
check against continuous sensitivities.

`DAEReplayPolicy("full")` stores the complete replay trajectory.
`DAEReplayPolicy("chunked", chunk_size=k)` rematerializes fixed-size segments during
the reverse pass. A chunk may instead be selected from `memory_budget_bytes`; exactly
one of `chunk_size` and `memory_budget_bytes` is required for chunked replay.
`DAESolvePlan` records the selected static chunk and conservative replay-memory
estimate, while `DAEReplayEvidence` records the realized accepted-step count. A failed
adaptive primal has no valid derivative.

## Local regularity evidence

`DAERegularityPolicy("solver-evidence")` records rank and convergence information
already available from consistency and nonlinear solves. The `"periodic"` mode
explicitly probes the configured consistency-coordinate Jacobian and the local
implicit-stage operator `F_y + shift * F_ydot` every `interval` accepted steps. Optional
`condition_limit` classifies an otherwise full-rank operator as numerically singular.
`failure="record"` preserves evidence without changing the solve; `"status"` promotes
a probed singular stage to an explicit rejected attempt and terminal status.

This evidence is local and numerical. It never claims a global differentiation index
or regularity between probes.

`DifferentialAlgebraicSolution` stores requested states, reconstructed stage rates,
node validity, accepted-step and attempt histories, residual and constraint norms,
initialization, continuation, local regularity, replay, termination, and
plan/method/linear-plan provenance. If initialization or integration fails, unsaved
nodes are `NOT_RUN`; no fallback state is fabricated. Use `failure="error"` when a
non-successful solution must raise at the call boundary.

## Semidiscrete implicit PDE residuals

`compile_semidiscrete_dae` lowers validated `PDEProblemIR` into a
`CompiledDiscreteResidual` and `DifferentialAlgebraicSystem`. It shares field layout,
coordinate, boundary-lift, parameter-binding, and expression validation with
`compile_semidiscrete_pde`, but retains temporal derivatives in the implicit
residual.

The required `equation_targets` mapping is a bijection from every equation name to
every field name. It determines residual component order and prevents positional
matching from silently changing the model. A differential target must contain exactly
one direct first temporal derivative of its target field. Algebraic targets contain
none. Temporal derivatives nested inside nonlinear or composite expressions are
rejected rather than approximated.

`SemidiscreteDAEStructuralReport` records targets, component roles, derivative counts,
and the explicit assumption `regular-index-1-required-unverified`. This is incidence
evidence, not a numerical rank certificate or an index claim. The compiled object's
`rate_jacobian` materializes the dense derivative of the residual with respect to the
state rate for diagnostics; the solver itself retains matrix-free native derivative
policies unless configured otherwise.

```text
compiled = phx.equations.compile_semidiscrete_dae(
    pde_problem,
    spatial_discretization,
    equation_targets={"momentum": "velocity", "constraint": "pressure"},
    parameter_values={"viscosity": 0.01},
)
dae_problem = phx.solver.DifferentialAlgebraicProblem(
    compiled.system,
    compiled.layout.pack(initial_fields),
    args=runtime_parameters,
)
```

Only semidiscrete states whose time derivative is representable by the PDE IR are
accepted. Unsupported temporal structure and a non-bijective residual layout fail at
compilation rather than becoming an invalid numerical solve.

## Trajectory interoperability

`phydrax.dynamics.identification.trajectory_data_from_differential_solution` accepts a
`DifferentialAlgebraicSolution`. It uses accepted BDF state rates as trajectory
derivatives, preserves componentwise rate validity, derives transition validity from
adjacent node validity, and retains the DAE plan in the source ID. The initialization
node is not labeled derivative-valid when the chosen consistency contract did not
determine all algebraic rates.

## API reference

::: phydrax.dynamics.DAEStructure

---

::: phydrax.dynamics.DifferentialAlgebraicSystem

---

::: phydrax.dynamics.HeldInputPolicy

---

::: phydrax.solver.DAEInitializationSpec

---

::: phydrax.solver.DAEInitializationResult

---

::: phydrax.solver.DAEInitializationStatus

---
::: phydrax.solver.DAEAdaptivePolicy

---

::: phydrax.solver.DAETemporalReusePolicy

---

::: phydrax.solver.DAEReplayPolicy

---

::: phydrax.solver.DAEReplayEvidence

---

::: phydrax.solver.DAERegularityPolicy

---

::: phydrax.solver.DAERegularityEvidence

---

::: phydrax.solver.DAERegularityStatus

---

::: phydrax.solver.DAEContinuation

---

::: phydrax.solver.DAEStepHistory

---

::: phydrax.solver.DAEAttemptHistory

---

::: phydrax.solver.DAEAttemptStatus

---

::: phydrax.solver.DAETerminationStatus

---


::: phydrax.solver.DAESolvePolicy

---

::: phydrax.solver.DAESolvePlan

---

::: phydrax.solver.PreparedDAESolve

---

::: phydrax.solver.DifferentialAlgebraicProblem

---

::: phydrax.solver.DifferentialAlgebraicSolution

---

::: phydrax.solver.DAEStatus

---

::: phydrax.solver.initialize_dae

---

::: phydrax.solver.plan_dae

---

::: phydrax.solver.prepare_dae

---

::: phydrax.solver.solve_dae

---

::: phydrax.equations.SemidiscreteDAEStructuralReport

---

::: phydrax.equations.CompiledDiscreteResidual

---

::: phydrax.equations.compile_semidiscrete_dae
