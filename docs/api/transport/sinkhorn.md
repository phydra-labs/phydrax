# Sinkhorn solving and divergence

`Sinkhorn` solves balanced entropic transport in stabilized log coordinates. The dense
and blockwise modes implement the same equations and return the same result contract.
Both `Sinkhorn` and `UnbalancedSinkhorn` default to `block_size="auto"` with
`sinkhorn_workspace_bytes=134217728` (128 MiB). This shared budget admits the
existing dense route when four scalar source-by-target arrays fit: cost, log
coupling, coupling, and reduction scratch. Full-fit execution uses exactly the
dense equations, without tile loops. In float64 the default admits square
problems through 2048 atoms per measure, covering the measured moderate
capacities (512, 1024, and 2048). This is a workspace choice, not a timing claim.

Above the budget, the square tile is
`clamp(floor(budget / (4 * itemsize * max(source_count, target_count))), 1, max(source_count, target_count))`.
The conservative linear extent bound also bounds each tile's four scalar pair
arrays. At least one pair is always executed even if its minimum scratch exceeds
the budget. The budget is logical cost/kernel scratch, not a cap on total device
memory: inputs (including precomputed costs), outputs, custom ground-cost internals,
payload widths in plan actions, and autodiff storage are separate.

A positive integer `block_size` explicitly overrides automatic admission, even
for full-fit problems; `block_size=None` explicitly requests dense execution,
even above budget. The budget must still be a positive integer (booleans and
nonintegers are refused) for either override. Blockwise mode does not allocate
the complete cost or plan during solving, statistics, or plan application.
`TransportProvenance.block_size` and the result's `block_size` record the
executed tile, or `None` for dense execution, never the `"auto"` request.
`TransportProvenance.sinkhorn_workspace_bytes` records the configured budget
as static numerical identity, including when an explicit override was used.
Archive templates must retain the budget and actual execution route.

`early_stop=False` is the compiled-work default: all configured iterations run, while
diagnostics preserve the first converged check. `early_stop=True` short-circuits after
the convergence condition is met. `store_history=True` retains one residual per check;
otherwise the fixed-width history is empty.

::: phydrax.transport.Sinkhorn

## Results and matrix-free plan actions

::: phydrax.transport.SinkhornResult

---

::: phydrax.transport.SinkhornDiagnostics

---

::: phydrax.transport.TransportProvenance

---

::: phydrax.transport.require_converged

Use `apply_source_to_target` or `apply_target_to_source` for physical coupling actions.
Use the barycentric variants for conditional payload averages; they divide by the
receiving marginal. `dense_plan()` is explicit and requires source-by-target memory
even when the solve was blockwise.

## Debiased Sinkhorn divergence

The regularized transport objective is not zero when a measure is compared with
itself. `sinkhorn_divergence` evaluates the cross solve and both self solves and
returns their debiased combination while retaining every diagnostic:

```py
divergence = phx.transport.sinkhorn_divergence(problem, solver)
if not divergence.converged:
    # Inspect divergence.cross, source_self, and target_self independently.
    ...
```

::: phydrax.transport.SinkhornDivergenceResult

---

::: phydrax.transport.sinkhorn_divergence

## Repeated fixed reference

A prepared reference lowers one fixed target and validates its self solve once.
Subsequent evaluations still solve the source-to-target and source self terms. Reuse is
valid only for the retained cost, solver, encoder semantics, and mass tolerance.

::: phydrax.transport.PreparedSinkhornReference

---

::: phydrax.transport.prepare_sinkhorn_reference

---

::: phydrax.transport.sinkhorn_divergence_against

## Differentiation

The solver is written as ordinary JAX control flow, so `jax.grad`, `jax.jvp`, and
`jax.vjp` differentiate the executed finite iteration map. This is unrolled
algorithmic differentiation, not implicit differentiation of an exact optimizer.
Iteration count, stopping policy, block size, and regularization therefore belong to
the differentiated numerical contract. Failed convergence is not a meaningful loss;
use `require_converged` or an integration that enforces it.
