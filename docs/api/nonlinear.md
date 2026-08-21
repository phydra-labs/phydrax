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

## Fixed points and nonlinear acceleration

`FixedPointProblem` represents `state = mapping(state, args)`. `PicardIteration` and
`FixedPointIteration` provide unaccelerated relaxation. `AndersonAcceleration` solves
a regularized least-squares mixing problem with bounded history and restarts when the
mix is unusable. `NonlinearGMRES` constructs a residual-minimizing affine candidate,
checks it against the unaccelerated proposal, and restarts rather than accepting a
harmful acceleration.

The fixed-point residual is always the physical defect `mapping(state) - state`.
Convergence and diagnostics are based on this defect, not on coefficient norms in an
acceleration subproblem.

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
`SemismoothNewton` solves a Fischer--Burmeister complementarity residual with a
configurable `GeneralizedDerivativePolicy`. Infinite one-sided bounds are handled
without invalid arithmetic.

`ComplementarityCertificate` reports lower/upper feasibility, natural residual,
Fischer--Burmeister residual, active sets, and finiteness separately. A loose
nonlinear residual tolerance cannot turn an infeasible point into a successful VI
result; final success requires the complementarity certificate as well.

## Implicit root differentiation

`implicit_root` differentiates the mathematical solution map through the accepted
root using a fresh linearized solve. It does not differentiate through iteration
history. Singular or incompatible derivative systems fail according to the supplied
linear policy instead of returning an unverified gradient.

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

::: phydrax.nonlinear.NonlinearTransformationEvidence

---

::: phydrax.nonlinear.implicit_root
