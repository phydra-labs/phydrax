# Delay and functional differential equations

Phydrax represents causal delay equations with a declared memory contract and solves
them through Diffrax. A problem has the form

\[
\mathrm dY_t = f(t, Y_t, M_t, a)\,\mathrm dt
+ \sum_k g_k(t, Y_t, M_t, a)\,\mathrm dW_t^k,
\]

where each named entry of `M_t` is declared by a delay term. The declaration, rather
than an unstructured history callback inside the vector field, lets the solver enforce
causality, propagate discontinuities, select quadrature, and report capability-specific
provenance.

## Problem and memory contract

`DelayDifferentialProblem` receives:

- `drift(t, state, memory, args)`, where `memory` is a `DelayValues` object;
- `history(t, args)`, valid at and before `t0`;
- one or more uniquely named delay terms;
- optional `history_derivative`, Wiener terms, and state geometry.

A drift may address memory by static name or declaration-order index:

```python
import jax.numpy as jnp
import phydrax as phx

problem = phx.solver.DelayDifferentialProblem(
    lambda t, y, memory, args: -0.2 * y + memory["feedback"],
    lambda t, args: jnp.ones((2,)),
    (phx.solver.ConstantDelay("feedback", 0.4),),
    t0=0.0,
    t1=2.0,
)
```

Every history value, drift value, and delay observation has exactly `state_shape`.
Scalar, vector, matrix, and higher-rank array states are supported. Delay names and
term ordering are static PyTree structure; values and trainable parameters remain JAX
array leaves.

## Causal method of steps

`solve_diffrax_delay` wraps a compatible Diffrax solver. Every accepted step contributes
its native local interpolation to delay history. Rejected candidate steps never become
queryable. For a method whose largest stage abscissa is `c_max`, Phydrax enforces

\[
\Delta t \leq \tau_{\min} / c_{\max},
\]

where `tau_min` is the certified smallest queried lag. Adaptive controllers are capped;
fixed solves require `diffrax.ConstantStepSize()` and an explicit `dt0`. Known
constant-delay discontinuities are propagated additively and steps are shortened at
their locations.

```python
import diffrax as dfx

save_times = jnp.linspace(0.0, 2.0, 101)

adaptive = phx.solver.solve_diffrax_delay(
    problem,
    save_times=save_times,
    solver=dfx.Tsit5(),
    rtol=1e-6,
    atol=1e-8,
    dense=True,
)

fixed = phx.solver.solve_diffrax_delay(
    problem,
    save_times=save_times,
    solver=dfx.Heun(),
    stepsize_controller=dfx.ConstantStepSize(),
    dt0=0.01,
)
```

Implicit fixed-step methods require an explicitly tolerance-configured Optimistix root
finder, as in ordinary Diffrax. `BacksolveAdjoint` is not a retarded adjoint and is
rejected. Use `CheckpointedDelayAdjoint`, `RecursiveCheckpointAdjoint`, or
`DirectAdjoint`.

## Delay terms

### Constant point delay

`ConstantDelay(name, delay)` evaluates `Y(t - delay)`. Delays must be finite and
strictly positive. Multiple point delays may be mixed in declaration order.

### State-dependent point delay

`StateDependentDelay` evaluates `Y(t - tau(t, Y(t), args))`. It requires a certified
positive lower bound, accepts an optional upper bound, and declares whether the delayed
argument is monotone. The lower bound controls causal step sizes. High-order execution
tracks descendants of known discontinuities as roots of
`t - tau(t, Y(t), args) - source = 0`; declarations that violate their bounds fail at
runtime.

For nonmonotone delayed arguments, set `monotone_argument=False`. The tracker isolates
every sign-changing or tangent root in each accepted bracket by recursive subdivision,
then refines the ordered roots with the declared tolerances and iteration bound.

### Functional history delay

`FunctionalDelay` passes a `DelayHistoryWindow` to
`functional(t, state, history, args)`. The callback may make scalar or batched lag
queries anywhere in its declared interval and may request one-sided values. Optional
`discontinuity_lags` identify exact translations that must participate in the
discontinuity closure.

The upper lag bound may be `jnp.inf`. This declares genuine infinite prehistory and is
supported by full accepted history: the user-supplied `history` remains authoritative
for arbitrarily old times, while accepted local interpolants cover the solved interval.
Infinite-memory terms deliberately reject rolling and segmented execution because no
finite eviction horizon exists.

```python
lags = jnp.asarray([0.2, 0.6, 1.4])
tail = phx.solver.FunctionalDelay(
    "tail",
    lambda t, y, history, args: jnp.mean(history.values(lags), axis=0),
    (0.1, jnp.inf),
    discontinuity_lags=lags,
)
```

### Distributed delay

`DistributedDelay` represents

\[
M(t) = \int_{\tau_a}^{\tau_b}
K(t, s, Y(t), a)Y(t-s)\,\mathrm ds.
\]

It uses a declared fixed interval quadrature rule. All quadrature history queries are
batched. Kernels may be scalar or exactly state-shaped, and nodes must query strictly
positive lags. Quadrature family, order, node count, and effective lag bounds are
recorded in solution metadata.

### Delayed derivatives and transformed neutral equations

`DerivativeDelay` supplies a delayed history derivative inside an ordinary declared
memory vector field. A problem containing one must provide `history_derivative` for
prehistory. Accepted-history derivatives come from an analytic solver interpolant or a
JAX JVP of that interpolant; Phydrax never silently finite-differences dense output.
Non-Euclidean derivative terms require explicit tangent transport.

`NeutralDelayProblem` handles the more general transformed equation
`d(y - N_retarded - N_endpoint)/dt = F`. The retarded neutral functional receives
named constant-delay values. An optional endpoint-dependent neutral functional is
inverted by an Optimistix root solve at every stage and accepted endpoint. The
transformed state, physical state, recovery tolerances, and recovery solver are
reported explicitly. This family currently uses Euclidean states, constant point
delays, deterministic `diffrax.Euler`, and an explicit `dt0`.

Brownian neutral equations are rejected: a pathwise delayed state derivative is not
defined for Brownian trajectories.

## Stochastic delay equations

Stochastic coefficients are declared as `DelayWienerTerm` objects. Each term specifies
its coefficient, native noise shape, noise structure, and optional basis identity. A
single global `WienerRealization` supplies every increment so coupled solves, replay,
and common-random-number studies preserve path identity.

```python
import jax.random as jr

stochastic_problem = phx.solver.DelayDifferentialProblem(
    lambda t, y, memory, args: -y + memory["lag"],
    lambda t, args: jnp.ones((1,)),
    (phx.solver.ConstantDelay("lag", 0.25),),
    wiener_terms=(
        phx.solver.DelayWienerTerm(
            "forcing",
            lambda t, y, memory, args: jnp.full((1, 1), 0.1),
            (1,),
            structure="additive",
            basis_id="forcing-basis",
        ),
    ),
    interpretation="ito",
    t0=0.0,
    t1=1.0,
)

realization = phx.stochastic.WienerRealization(
    jr.key(0),
    (1,),
    support=(0.0, 1.0),
    noise_id="forcing-basis",
)

stochastic_solution = phx.solver.solve_diffrax_delay(
    stochastic_problem,
    save_times=jnp.linspace(0.0, 1.0, 101),
    solver=dfx.Euler(),
    stepsize_controller=dfx.ConstantStepSize(),
    dt0=0.01,
    realization=realization,
)
```

Certified Euclidean methods are fixed-step `diffrax.Euler` for Itô equations and
`diffrax.EulerHeun` for Stratonovich equations. Both require
`diffrax.ConstantStepSize`, an explicit `dt0`, and a compatible global realization.
Constant, distributed, functional, and state-dependent retarded terms share the same
path-consistent accepted history. State-dependent stochastic lags use the declared
positive bound but do not propagate random discontinuity roots path by path.

Adaptive stochastic controllers and arbitrary stochastic interpolations are rejected;
an SDE solver is not accepted merely because it exposes an `interpolation_cls`.

## Geometric delay equations

Set `state_geometry` on the problem and choose a compatible Phydrax geometric solver.
Fixed-step geometric local interpolations are retained in accepted history, so delayed
values and dense output stay on the declared manifold. Problem and solver geometry IDs
must agree. Non-Euclidean distributed terms require an explicit reducer, and derivative
terms require tangent transport.

Deterministic execution supports `GeometricEuler`, `RKMK`, and commutator-free methods.
Intrinsic Stratonovich delay equations use `SRKMK` with a path-consistent geometric
Wiener interpolation; intrinsic Itô geometry is rejected because Itô corrections are
not encoded by a retraction-only tangent update.

## Long-horizon bounded history

Causal initial-value problems with a finite maximum lag can bound active device memory.
For one compiled solve, set `history_mode="rolling"` and supply `history_capacity` for
adaptive or traced execution. Fixed-step eager execution can derive the exact ring
capacity from the maximum lag, `dt0`, and scheduled breakpoints.

```python
rolling = phx.solver.solve_diffrax_delay(
    problem,
    save_times=jnp.asarray([2.0]),
    solver=dfx.Heun(),
    stepsize_controller=dfx.ConstantStepSize(),
    dt0=0.01,
    history_mode="rolling",
)
```

`solve_diffrax_delay_segmented` additionally bounds each compiled Diffrax loop. The host
driver resumes exact solver, controller, discontinuity-tracker, neutral-recovery, and
Wiener-path state across windows. Rejected candidates never enter the rolling ring.
Setting `dense=True` archives completed interpolations on the host; without it, only
requested values and the live continuation state remain.

```python
segmented = phx.solver.solve_diffrax_delay_segmented(
    problem,
    save_times=jnp.asarray([2.0]),
    solver=dfx.Heun(),
    stepsize_controller=dfx.ConstantStepSize(),
    dt0=0.01,
    max_steps_per_segment=64,
)

continuation = segmented.continuation
active_history = continuation.active_history
```

The number of host windows is data-dependent, so ordinary segmented execution is not a
whole-solve JIT primitive. For differentiation, pass
`SegmentedDelayAdjoint(max_segments, checkpoints=...)`; tracing replays the numerically
identical controller and rolling-history state in one statically bounded loop and uses
recursive checkpointing. Dense output and continuation input are intentionally excluded
from that replay contract. Infinite-memory functionals reject both rolling and
segmented modes.

## Rough delay equations

`RoughDelayDifferentialProblem` integrates a geometric rough differential equation
against an `AbstractRoughControl`. `RoughEuler` uses first-level increments. `Davie`
also consumes second-level iterated integrals and includes both current and delayed
directional derivatives. Constant point delays must align with control partition nodes
for the Davie cross-level contract.

```python
rough_times = jnp.linspace(0.0, 1.0, 21)
rough_control = phx.stochastic.GeometricRoughPath.from_values(
    rough_times, rough_times[:, None]
)
rough_problem = phx.solver.RoughDelayDifferentialProblem(
    lambda t, y, memory, args: jnp.ones(y.shape + (1,)),
    lambda t, args: jnp.ones((1,)),
    (phx.solver.ConstantDelay("lag", 0.2),),
    t0=0.0,
    driver_dimension=1,
    drift=lambda t, y, memory, args: -0.1 * y + memory["lag"],
)
rough_solution = phx.solver.solve_rough_delay(
    rough_problem,
    rough_control,
    solver=phx.solver.Davie(),
)
```

The control records one global rough path and may carry batch axes. State geometry is
enforced by tangent projection and retraction at every update. Metadata identifies
control depth, delayed second-level policy, interpolation, and prehistory enhancement.

## Prescribed finite-activity jump delays

`JumpDelayProblem` wraps a retarded `DelayDifferentialProblem` with
`jump(time, pre_state, memory, channel, mark, args)`. `solve_jump_delay` consumes one
successful, unbatched `JumpEventBatch`, solves every open interval with exact dense
delay history, and applies jumps at their prescribed interior times. The resulting
history is right-continuous; dense evaluation with `left=True` recovers the pre-jump
state. Optional Wiener dynamics reuse one global, unbatched `WienerRealization` across
all intervals.

This backend is for exogenous event schedules. It does not reinterpret a
state-dependent jump intensity as prescribed times, and derivative-valued delay terms
are rejected because a jump derivative requires a separate distributional contract.

## Convolution and fractional memory equations

`ConvolutionVolterraProblem` specializes `StochasticVolterraProblem` to
translation-invariant kernels `kernel(target - source, args)`.
`solve_convolution_volterra` performs causal left-point accumulation for deterministic
or Wiener-driven equations and preserves global realization identity.

`CaputoFractionalProblem` declares `D_C^alpha y = f(t, y, args)` for
`0 < alpha <= 2`. Orders above one require `initial_derivative`.
`solve_caputo_fractional` uses exact power-law cell weights with explicit
piecewise-constant product integration. Nonuniform grids, JIT, and reverse-mode
differentiation are supported; the generic backend uses quadratic memory work and
linear state storage.

## Global future/past equations

Equations that depend on future state are not causal initial-value problems.
`solve_functional_differential` instead treats the whole trajectory as a piecewise
Chebyshev--Lobatto polynomial unknown and solves differential, continuity, boundary,
periodic, phase, and observation residuals by global Optimistix root finding or least
squares. Every interval uses ascending endpoint-inclusive reference nodes, analytic
barycentric interpolation weights, and Clenshaw--Curtis weights for the unweighted
Lebesgue residual measure.

```python
advanced_problem = phx.solver.FunctionalDifferentialBoundaryProblem(
    lambda t, y, values, args: 2.0 * values[0] - t,
    argument_times=lambda t, y, args: jnp.asarray([0.5 * (t + 1.0)]),
    num_arguments=1,
    boundary=lambda left, right, trajectory, args: left,
)
plan = phx.solver.FunctionalCollocationPlan(
    jnp.asarray([0.0, 0.25, 0.7, 1.0]),
    degree=2,
)
advanced_solution = phx.solver.solve_functional_differential(
    advanced_problem,
    plan,
    lambda t, args: t,
)
```

Declaring `parameter_shape` adds physical parameters to the nonlinear unknown PyTree;
provide their initial value with `parameter_guess`. Declaring `unknown_period=True`
requires a periodic problem and an explicit phase constraint; provide a positive
`period_guess`. The period is solved in log coordinates and maps the plan's reference
mesh to physical time. Callbacks then receive
`FunctionalDifferentialContext(args, parameters, period)`, and the solution exposes
the inferred `parameters` and `period`. With no inferred quantities, callbacks receive
the original `args` unchanged.

## Capability matrix

| Equation class | Execution | Required numerical contract |
| --- | --- | --- |
| Constant, distributed, or functional retarded; deterministic | Fixed or adaptive Diffrax | Transformable local interpolation; positive causal lag bound |
| Infinite-memory functional; deterministic or Wiener | Full accepted history | User prehistory on the full negative-time domain; finite queried lags |
| State-dependent retarded; deterministic | Tracked method of steps | Positive certified lag bound; dynamic-root and discontinuity capacities |
| Retarded Wiener equation | Fixed Euler/Euler--Heun | Global Wiener realization; path-consistent accepted interpolation |
| Intrinsic Stratonovich retarded equation | Fixed `SRKMK` | Matching geometry; tangent diffusion; geometric Wiener interpolation |
| Geometric retarded; deterministic | Fixed geometric methods | Matching geometry; manifold-valued interpolation; reducer/transport where required |
| Transformed neutral; deterministic | Fixed Euler | Constant point delays; Euclidean state; recovery root contract |
| Long-horizon finite-memory equation | Rolling or segmented | Finite maximum lag; adequate active-history capacity |
| Rough retarded equation | RoughEuler or Davie | Geometric rough control; Davie delay/partition alignment |
| Prescribed jump-delay equation | Host hybrid Diffrax intervals | Successful unbatched event stream; interior, strictly ordered times |
| Translation-invariant Volterra equation | Direct causal convolution | Explicit left accumulation; optional global Wiener realization |
| Caputo fractional equation | Product integration | Order in `(0, 2]`; initial derivative above one |
| Advanced or mixed future/past equation | Global functional collocation | Finite represented interval; boundary/periodic/phase constraints |

Explicit limitations are numerical contracts, not silent fallbacks. Phydrax rejects
Brownian neutral equations, intrinsic Itô manifold execution, adaptive Euclidean delay
SDE controllers, arbitrary stochastic interpolations, `BacksolveAdjoint`,
infinite-memory rolling/segmented execution, manifold distributed terms without a
reducer, and manifold derivative terms without tangent transport. Segmented host
execution does not support batched Wiener realizations or whole-solve JIT with an
unknown window count. Pathwise stochastic state-dependent delays enforce their declared
lag bounds but do not track one random discontinuity-root tree per sample.

The rough Davie backend currently requires constant partition-aligned point delays.
The jump-delay backend accepts prescribed unbatched interior events and excludes
derivative-valued delays; state-dependent jump intensities remain the responsibility of
a jump-process solver rather than being reinterpreted as fixed event times.

## Reproducible benchmark

```console
uv run python tools/delay_benchmarks.py --repeats 10
```

The JSON report separates lowering/compilation (`compile_ms`), first execution, and
steady execution. Its baseline compares ordinary and retarded smooth/stiff Diffrax
solves. Family records exercise functional and transformed-neutral method-of-steps,
Davie rough delay, prescribed jumps, translation-invariant Volterra convolution, and
Caputo product integration. Both whole-solve rolling and host-segmented short/long
horizon records expose history capacity and allocated bytes, testing bounded active
memory independently of total horizon.

The prescribed-jump record constructs one immutable `JumpDelayProblem` and reuses it
across timing repeats. Its first-execution time includes compilation of the continuous
interval shapes; steady time reuses those executables. Reconstructing Python callables
inside a solve loop creates distinct static JAX programs and recompiles them.

## Result and provenance

Causal Diffrax, convolution, fractional, and jump-delay solves return
`MemoryEquationSolution`. It records saved values and validity, accepted/rejected work,
controller and causal-step policy, history occupancy, discontinuity tracking,
delay-term and functional contracts, geometry, quadrature or memory-kernel metadata,
driver/realization identity, solver ID, and resolved method. `solve_rough_delay` returns
`RoughDifferentialSolution`, with interval statuses, rough-control statistics, and the
same solver/geometry provenance. Dense evaluation is available only when explicitly
requested and only over the archived solved interval.

## API

::: phydrax.solver.DelayDifferentialProblem

---

::: phydrax.solver.DelayValues

---

::: phydrax.solver.DelayHistoryWindow

---

::: phydrax.solver.ConstantDelay

---

::: phydrax.solver.StateDependentDelay

---

::: phydrax.solver.FunctionalDelay

---

::: phydrax.solver.DistributedDelay

---

::: phydrax.solver.DerivativeDelay

---

::: phydrax.solver.NeutralDelayProblem

---

::: phydrax.solver.DelayWienerTerm

---

::: phydrax.solver.CheckpointedDelayAdjoint

---

::: phydrax.solver.SegmentedDelayAdjoint

---

::: phydrax.solver.MemoryEquationSolution

---

::: phydrax.solver.solve_diffrax_delay

---

::: phydrax.solver.fixed_delay_history_capacity

---

::: phydrax.solver.DelaySegmentContinuation

---

::: phydrax.solver.DelaySegmentArchive

---

::: phydrax.solver.SegmentedDelayResult

---

::: phydrax.solver.solve_diffrax_delay_segmented

---

::: phydrax.solver.RoughDelayDifferentialProblem

---

::: phydrax.solver.solve_rough_delay

---

::: phydrax.solver.JumpDelayProblem

---

::: phydrax.solver.JumpDelayBackendResult

---

::: phydrax.solver.solve_jump_delay

---

::: phydrax.solver.ConvolutionVolterraProblem

---

::: phydrax.solver.solve_convolution_volterra

---

::: phydrax.solver.CaputoFractionalProblem

---

::: phydrax.solver.solve_caputo_fractional

---

::: phydrax.solver.FunctionalDifferentialContext

---

::: phydrax.solver.FunctionalDifferentialBoundaryProblem

---

::: phydrax.solver.FunctionalCollocationPlan

---

::: phydrax.solver.FunctionalDifferentialSolution

---

::: phydrax.solver.solve_functional_differential

## Bounded stochastic adaptation, adjoints, and long memory

Stochastic-delay interpolation is capability typed.
`ItoEulerDelayInterpolation`, `StratonovichEulerHeunDelayInterpolation`, and
`SRKMKDelayInterpolation` declare interpretation, strong order, causal replay,
Levy-area, geometry, and noise-structure requirements. Untyped arbitrary Diffrax
interpolation remains unsupported. `adaptive_stochastic_delay_step_doubling`
compares one full step with two half steps on the identical Brownian increments,
accepts only the two-half history, and records fixed-capacity attempt/accept
evidence. Rejection never resamples or enters causal history.

`BacksolveDelayAdjoint` is not `diffrax.BacksolveAdjoint`. It integrates the
continuous advanced adjoint only for smooth deterministic retarded constant point
delays (and statically represented quadrature terms) from a matching
`DelayPrimalTape`. It is a convergent continuous-adjoint approximation, not the
exact checkpointed discrete gradient and not constant-memory. Unsupported neutral,
state-dependent, functional, stochastic, manifold, or hybrid terms fail
preparation.

Bounded infinite memory has two truthful forms.
`ExponentialConvolutionDelay` carries an exact finite-dimensional declared
sum-of-exponentials realization. `CertifiedTruncatedFunctionalDelay` evaluates a
finite retained window only while its conservative omitted-tail bound is finite,
nonnegative, and below tolerance. Arbitrary `FunctionalDelay(..., upper=inf)`
without either representation still requires full history.

An explicit `FixedCapacitySegmentPolicy` on
`solve_diffrax_delay_segmented` selects the whole-solve fixed-shape route and
emits `FixedCapacitySegmentEvidence`. Unknown unbounded segment counts retain the
host-streaming route. Reaching a segment/step/event cap is failure, never partial
success.

::: phydrax.solver.AdaptiveStochasticDelayPolicy

---

::: phydrax.solver.BacksolveDelayAdjoint

---

::: phydrax.solver.ExponentialConvolutionDelay

---

::: phydrax.solver.CertifiedTruncatedFunctionalDelay

---

::: phydrax.solver.FixedCapacitySegmentPolicy
