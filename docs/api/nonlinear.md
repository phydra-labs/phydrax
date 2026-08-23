# Nonlinear systems

## Scope and contracts

`phydrax.nonlinear` owns nonlinear algebraic systems, fixed-point iterations,
nonlinear preconditioning, full-approximation multigrid, variational inequalities,
and implicit root derivatives. Optimization remains in `phydrax.optim`; continuation
and bifurcation workflows remain in `phydrax.continuation`.

A root workflow has four explicit pieces:

1. `NonlinearSystemProblem` defines the residual, optional auxiliary output, and
   accepted-state validity predicate.
2. `JacobianPolicy` selects automatic JVP/VJP actions, sparse derivatives,
   directional finite differences, or an explicit operator supplied by the caller.
3. an `AbstractNonlinearMethod` supplies iteration and globalization semantics.
4. `NonlinearTermination` supplies residual, step, work, and divergence limits.

The returned `NonlinearResult` always carries a portable status, residual, numerical
work diagnostics, capabilities, and provenance. A finite trial rejected by a validity
predicate is distinct from a nonfinite evaluation. A failed inner linear solve is
also distinct from line-search or trust-region rejection.

`NonlinearTermination.maximum_linear_iterations` is a hard aggregate inner-work
budget. Newton methods pass the remaining allowance into each prepared native
Krylov solve, so the last admissible solve cannot overrun the outer contract.
`NewtonForcingPolicy` chooses a constant or Eisenstat--Walker relative tolerance
without rebuilding the prepared linear plan. `JacobianRefreshPolicy` independently
selects every-step, periodic, stagnation, or globalization-rejection refresh.

!!! example
    ```python
    import jax.numpy as jnp
    import phydrax as phx


    problem = phx.nonlinear.NonlinearSystemProblem(
        lambda state, args: {
            "temperature": state["temperature"] ** 2 - args,
        },
        problem_id="positive-square-root",
        validity=lambda state, residual, auxiliary, args: (
            state["temperature"] >= 0.0
        ),
    )
    result = phx.nonlinear.root(
        problem,
        {"temperature": jnp.asarray(1.0)},
        method=phx.nonlinear.NewtonKrylov(),
        termination=phx.nonlinear.NonlinearTermination(
            absolute_residual=1e-10,
            relative_residual=0.0,
        ),
        args=jnp.asarray(2.0),
    )
    assert bool(result.successful)
    ```

## Newton methods and linear coordinates

`NewtonKrylov` uses a residual-merit Armijo search. `NewtonTrustRegion` accepts a
step through agreement between the physical residual norm and its local linear
model. Both reuse the symbolic linear-solve plan across accepted Jacobian refreshes.
Their default Jacobian is matrix-free and their default linear method is restarted
GMRES.
When the linear policy declares `RecyclingPolicy`, Newton carries the immutable
GCRO-DR state between inner solves and refreshes retained operator images after
numeric Jacobian updates.

A nonlinear state and residual may use different PyTree containers while still having
the same coordinate dimension. Phydrax then rebases the Jacobian action onto one
canonical coordinate space while preserving its matrix-free JVP/VJP actions. It does
not materialize a dense Jacobian or change the requested linear method. Unequal
coordinate dimensions are rejected; large structured problems may instead expose
compatible state/residual spaces or supply a deliberate sparse or preconditioned
linear policy.

`RootLineSearch` and `RootTrustRegion` expose every globalization constant. Domain
rejections, nonfinite trials, accepted/rejected steps, Jacobian preparations, linear
iterations, and refresh counts are returned in `NonlinearDiagnostics`.

For repeated solves with unchanged spaces and derivative structure, split symbolic
selection from numerical refresh:

```python
prepared = phx.nonlinear.prepare_nonlinear(
    problem,
    {"temperature": jnp.asarray(1.0)},
    method=phx.nonlinear.NewtonKrylov(),
    args=jnp.asarray(2.0),
)
first = phx.nonlinear.solve_prepared_nonlinear(prepared)

changed = phx.nonlinear.NonlinearSystemProblem(
    problem.residual_function,
    validity=problem.validity_function,
    problem_id=problem.problem_id,
)
refreshed = phx.nonlinear.refresh_nonlinear(
    prepared,
    changed,
    {"temperature": jnp.asarray(1.0)},
    args=jnp.asarray(3.0),
)
second = phx.nonlinear.solve_prepared_nonlinear(refreshed)
```

Refresh preserves the linear template and plan IDs while replacing the residual,
Jacobian actions, initial state, and numeric linear state. Changed problem identity,
spaces, derivative structure, or unsupported nonlinear methods are rejected rather
than replanned. Per-call termination overrides may tighten work limits without
changing the prepared artifact.

## Finite nonlinear updates and composition

A complete solve and a finite update have different contracts. An
`AbstractNonlinearUpdate` may apply a Newton step, Picard correction, FAS cycle,
or subdomain sweep without claiming that the resulting state is a root.
`NonlinearUpdateResult.applied` means that bounded work produced a finite,
physically valid proposal; only a complete `NonlinearResult` can certify
convergence.

`plan_nonlinear_update`, `prepare_nonlinear_update`,
`refresh_nonlinear_update`, and `apply_prepared_nonlinear_update` bind the
physical state/residual spaces and retain update history or prepared linear
state. Work controls are checked before an indivisible update. Arbitrary
callables enter this lifecycle only through `FunctionNonlinearUpdate`.

`CompositeNonlinearUpdate` supports static multiplicative, weighted additive,
and safeguarded residual-optimal composition. Every child is evaluated from
the declared base state, its result remains component evidence, and the
combined candidate is independently evaluated against the original physical
problem. `NonlinearRichardson` adds physical-residual Armijo globalization.
`NonlinearGMRES` accepts a typed update rather than an untracked callable.

`NonlinearSubdomain` declares state and residual restrictions, correction
prolongation, one local residual, and a typed local update.
`NonlinearAdditiveSchwarz`, `NonlinearMultiplicativeSchwarz`, and
`NonlinearGaussSeidel` preserve this topology under JIT and certify the global
residual after local work. `ASPIN` uses additive Schwarz as a left nonlinear
preconditioner and prepares local dense Jacobian inverses for reuse within each
outer approximate-Jacobian action. Its nested local work is explicitly marked
as incompletely countable.

## Fixed points and nonlinear acceleration

`FixedPointProblem` represents `state = mapping(state, args)`. `PicardIteration`
and `FixedPointIteration` provide unaccelerated relaxation.
`AndersonAcceleration(kind="type-i" | "type-ii")` uses a fixed-capacity
regularized history solve with conditioning, residual-growth, and coefficient
safeguards. `SteffensenIteration` provides elementwise Aitken/Steffensen
acceleration and falls back to the ordinary mapped point when the accelerated
residual is worse or the denominator is singular.

Fixed-point termination is the physical norm of `mapping(state) - state`.
Convergence of an acceleration coefficient solve is never treated as fixed-point
convergence.

## Nonlinear preconditioning

Left and right nonlinear preconditioners have separate contracts:

- a left preconditioner maps a physical residual into a declared residual space;
- a right preconditioner maps a latent state into the physical state before residual
  evaluation.

`LeftPreconditionedSystem` and `RightPreconditionedSystem` retain both coordinate
spaces and reconstruct the physical result explicitly. Phydrax validates source,
target, state, and residual structures before iteration; it does not infer an
isomorphism from equal flattened sizes.

Every transformed solve returns the reconstructed physical state, physical residual,
physical auxiliary output, and a separate `NonlinearTransformationEvidence` record.
Success therefore certifies the original problem. A transformed residual may not
hide a nonfinite or invalid reconstructed physical root.

## Full approximation scheme

`FASHierarchy` is an immutable sequence of `FASLevel` objects. Every level declares
its nonlinear operator, restriction, prolongation, smoother, and coarse solve.
`FASCyclePolicy` selects V, W, or F traversal with explicit visit counts. `fas_cycle`
returns residual reduction, level visits, coarse solves, transfer counts, and a
portable status. `FASNonlinearPreconditioner` exposes the same hierarchy through the
nonlinear-preconditioner interface.

FAS uses the full-approximation coarse equation, including the tau correction. It is
not a linear correction scheme with nonlinear labels.

## Variational inequalities and complementarity

`VariationalInequalityProblem` combines a nonlinear map with explicit `Bounds`.
`SemismoothNewton` solves a natural-map or Fischer--Burmeister residual with a
configurable `GeneralizedDerivativePolicy`. Infinite one-sided bounds are
handled without invalid arithmetic.

`feasibility="allow-infeasible"` retains the ordinary semismooth Newton path.
`feasibility="preserve-box"` projects every trial before operator evaluation,
so an operator whose mathematical domain is the box never receives an
out-of-domain state. The accepted and returned state is exactly projected.

`prepare_variational_inequality`, `refresh_variational_inequality`, and
`solve_prepared_variational_inequality` separate compilation from repeated
solves. Refresh permits new finite bound values but rejects changed
unbounded/lower/upper/box/fixed topology.

`ComplementarityCertificate` reports lower/upper feasibility, natural residual,
Fischer--Burmeister residual, active sets, and finiteness separately. A loose
nonlinear residual tolerance cannot turn an infeasible point into a successful
VI result; final success requires the complementarity certificate as well.

## General root selection

Scalar equations use `ScalarRootProblem`. Certified bracket methods include
`Bisection`, `Brent`, `Ridder`, and `TOMS748`; safeguarded Newton and Halley
retain the bracket while consuming analytic or automatic derivatives.

Vector systems expose complementary local and global strategies:

| Problem evidence | Recommended method |
|---|---|
| small, smooth, accurate second derivatives | `VectorHalley` |
| smooth square system | `NewtonKrylov` or `NewtonTrustRegion` |
| repeated solve with stable Jacobian | `Chord` |
| expensive or unavailable Jacobian | `Broyden` or `DFSANE` |
| difficult basin or ill-conditioned start | `PseudoTransient` |
| deterministic fallback graph | `RobustRoot` |
| one capability-selected native attempt | `FastRoot` |

`RootPolyalgorithm` shares one physical work budget across attempts, retains
every attempt in `NonlinearResult.attempts`, and hands the best physical state
forward. The first selected method owns the initial residual evaluation.
Subsequent Broyden, pseudo-transient, and DF-SANE attempts consume the retained
residual and auxiliary value without evaluating them again. Consecutive Newton
attempts additionally retain the prepared Jacobian, symbolic linear template,
numeric refresh state, and Krylov recycling state; attempt-local counters and
globalization controls are reset before execution. Provenance reports residual
reuse and prepared handoff counts. `FastRoot` likewise selects from the state
dimension before residual preparation, so capability selection does not add a
hidden evaluation. Fallbacks are never hidden.

`NonlinearWorkBudget` carries traced residual, validity, derivative, setup,
linear, preconditioner, and local-update limits. `NonlinearAttemptEvidence`
forms a fixed-topology component tree. Multiplicative compositions truly
short-circuit: later components return zero-work skipped evidence and perform
no residual or local-solver work.

## Scaling, precision, batching, and sharding

`NonlinearScalingPolicy` prepares positive state and residual scaling while
retaining physical-unit certification. `NonlinearPrecisionPolicy` separates
model, direction, linear, and certificate precision;
`MixedPrecisionRootExecution` always re-evaluates the original problem in
certificate precision.

`SmallRootKernel` is a fixed-work, masked, batched dense-Newton kernel for small
array systems. Completed batch members stop updating independently.
`ShardedNonlinearPolicy` declares state/residual placement and whether norms are
local or collective; distributed semantics are never inferred from an array
that merely happens to be sharded.

## Solution-map derivatives

`SensitivityPolicy` selects implicit forward/reverse, unrolled, truncated, or
direct-loss-minimization semantics. `root_solution_jvp` and
`root_solution_vjp` solve the final physical linearization;
`root_solution_second_jvp` applies the second-order implicit-function formula
without constructing a third-order tensor. Singular, ill-conditioned, and
nonfinite derivative systems return explicit `SensitivityEvidence`.

## Implicit root differentiation

`implicit_root_result` runs the native nonlinear solve exactly once and returns a
`NonlinearResult` whose state, residual, and auxiliary output are evaluated at the
same accepted root. The state and user auxiliary leaves carry implicit JVP/VJP rules;
status, iteration counts, work diagnostics, provenance, and transformation evidence
remain nondifferentiable solver evidence. The explicit result therefore supports
differentiable downstream observables without discarding failure information.

`implicit_root` is the strict convenience form. It returns only the accepted state and
raises when the native solve is unsuccessful. It does not hide or repair failure; use
`implicit_root_result` when status-valued control flow or diagnostics are required.

Both entry points differentiate the mathematical root map through a fresh linearized
solve, not through nonlinear iteration history. They accept either an ordinary
problem/state/method contract or a `PreparedNonlinearSolve`, so a caller can retain one
symbolic linear template while refreshing runtime parameters. Singular or
incompatible derivative systems fail according to the supplied linear policy instead
of returning an unverified gradient.

## Causal nonlinear recurrence

`CausalRecurrenceProblem` represents a fixed-length first-order recurrence as the
direct residual `state[t] - transition(parameters, state[t - 1], driver[t])`.
`solve_causal_recurrence` evaluates all local transitions concurrently and solves
the resulting affine temporal system with associative scans. The core contract is
one trajectory; consumers use `vmap` for physical cases.

`CausalNewton` supplies exact dense DEER or an explicitly identified quasi-Newton
linearization. `CausalLevenbergMarquardt` supplies ELK-style damping with
actual-versus-predicted reduction checks. Dense, diagonal, fixed-block, and
fixed-probe Hutchinson diagonal policies are available through
`CausalLinearizationPolicy`. Approximate linearizations alter only the proposed
direction: success is always certified by the direct recurrence residual.

The returned histories distinguish convergence, stagnation, nonfinite evaluation,
damping rejection, and exhausted work. Maximum steps never imply success.
The solved trajectory carries an exact implicit derivative: the backward pass solves
the exact reverse block-bidiagonal recurrence at the converged state and does not
differentiate through iteration counts, damping choices, or quasi-Newton probes.
Differentiating a failed solve raises instead of returning an approximate gradient.

::: phydrax.nonlinear.CausalRecurrenceProblem

---

::: phydrax.nonlinear.CausalLinearizationPolicy

---

::: phydrax.nonlinear.CausalNewton

---

::: phydrax.nonlinear.CausalLevenbergMarquardt

---

::: phydrax.nonlinear.CausalRecurrenceResult

---

::: phydrax.nonlinear.solve_causal_recurrence

## API reference

::: phydrax.nonlinear.NonlinearSystemProblem

---

::: phydrax.nonlinear.NonlinearTermination

---

::: phydrax.nonlinear.NonlinearResult

---

::: phydrax.nonlinear.JacobianPolicy

---

::: phydrax.nonlinear.NewtonForcingPolicy

---

::: phydrax.nonlinear.JacobianRefreshPolicy

---

::: phydrax.nonlinear.PreparedNonlinearSolve

---

::: phydrax.nonlinear.prepare_nonlinear

---

::: phydrax.nonlinear.refresh_nonlinear

---

::: phydrax.nonlinear.solve_prepared_nonlinear

---

::: phydrax.nonlinear.NewtonKrylov

---

::: phydrax.nonlinear.NewtonTrustRegion

---
::: phydrax.nonlinear.AbstractNonlinearUpdate

---

::: phydrax.nonlinear.NonlinearUpdateResult

---

::: phydrax.nonlinear.PreparedNonlinearUpdate

---

::: phydrax.nonlinear.FunctionNonlinearUpdate

---

::: phydrax.nonlinear.NewtonStepUpdate

---

::: phydrax.nonlinear.CompositeNonlinearUpdate

---

::: phydrax.nonlinear.NonlinearRichardson

---

::: phydrax.nonlinear.NonlinearSubdomain

---

::: phydrax.nonlinear.NonlinearAdditiveSchwarz

---

::: phydrax.nonlinear.NonlinearMultiplicativeSchwarz

---

::: phydrax.nonlinear.NonlinearGaussSeidel

---

::: phydrax.nonlinear.ASPIN

---


::: phydrax.nonlinear.FixedPointProblem

---

::: phydrax.nonlinear.PicardIteration

---

::: phydrax.nonlinear.AndersonAcceleration

---

::: phydrax.nonlinear.NonlinearGMRES

---

::: phydrax.nonlinear.FASHierarchy

---

::: phydrax.nonlinear.FASCyclePolicy

---

::: phydrax.nonlinear.VariationalInequalityProblem

---

::: phydrax.nonlinear.SemismoothNewton

---

::: phydrax.nonlinear.PreparedVariationalInequalitySolve

---

::: phydrax.nonlinear.prepare_variational_inequality

---

::: phydrax.nonlinear.refresh_variational_inequality

---

::: phydrax.nonlinear.solve_prepared_variational_inequality

---


::: phydrax.nonlinear.NonlinearTransformationEvidence

---
::: phydrax.nonlinear.implicit_root_result

---


::: phydrax.nonlinear.implicit_root
