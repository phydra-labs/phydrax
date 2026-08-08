# Optimization

## Ownership and backend model

`phydrax.optim` is the sole namespace for optimization algorithms and configurations
owned by Phydrax. It is intentionally narrow: each object exposes a concrete numerical
contract, while the consuming workflow retains its geometry, sampling, objective,
reconstruction, validity, checkpoint, and result semantics.

Phydrax does not mirror upstream Optax, Evosax, or Optimistix APIs. Import those objects
from their native packages. A workflow accepts an external optimizer only when it has an
explicit adapter for that optimizer family.

## Structured residual optimization: KFAC

`phydrax.optim.kfac(...)` constructs the Phydrax-native structured optimizer used by
`FunctionalSolver.solve`. It targets nonnegative least-squares residual objectives and
uses a type-II generalized Gauss–Newton (GGN) curvature model.

!!! example
    ```python
    import phydrax as phx

    def train_with_kfac(solver):
        optimizer = phx.optim.kfac(
            damping=1e-3,
            factor_decay=0.95,
            factor_update_period=1,
            factor_chunk_size=32,
            approximation="expand",
            cg_max_steps=50,
            line_search=True,
        )
        return solver.solve(
            num_iter=100,
            optim=optimizer,
            seed=0,
            keep_best=True,
        )
    ```

### Curvature semantics

For square-root-weighted residual roots $r(\theta)$, KFAC models the type-II GGN
$J_r^\mathsf{T}J_r$. The curvature therefore remains nonzero when a residual value is
zero but its parameter Jacobian is not. Term scales, pointwise weights, masks, quadrature
masses, and mean/integral normalization are included exactly once when the residual roots
are formed.

Each supported affine weight-and-bias block is approximated independently. Contributions
from distinct `phydrax.terms.ResidualPenalty` terms retain separate
exponential-moving-average state and are combined as a sum of Kronecker products; no
artificial cross-term products are introduced. `approximation="expand"` retains every
singular mode of each
residual-event block Jacobian. `"reduce"` retains its leading mode and uses less work at
the cost of a coarser factorization. Both variants preserve the exact block trace.

Factor extraction differentiates bounded residual chunks with respect to one parameter
block at a time. The training path never constructs a residual-by-all-parameters
Jacobian. `factor_chunk_size` controls that memory/computation tradeoff. The summed
Kronecker operator is applied matrix-free and solved with diagonal-preconditioned
conjugate gradients. Damping is added once to the combined block operator.

### Differential residuals

The residual graph is differentiated as written, including hard-enforcement ansätze and
couplings between fields. First derivatives, pure and mixed second derivatives, and
contracted Laplacians are supported. Consequently, curvature includes parameter
sensitivities of the field derivatives, rather than treating derivative values as
ordinary activations.

Derivative requests above second order fail during optimizer setup. This boundary is
intentional: unsupported derivative curvature is never silently replaced with
first-order KFAC.

### Sampling and line search

Every optimizer step materializes one integration realization for each active term. The
same realizations—including adaptive-collocation weights—evaluation keys, and iteration
value are reused for the gradient, factor update, and all Armijo candidates. Adaptive
collocation is refreshed at most once before the step. When `train_term_sample_size` is
set, the first step warms all per-term factors; later steps update only the sampled terms
while retaining their stored factors.

`line_search=True` uses frozen-batch Armijo backtracking. A failed finite search returns a
zero step; a search whose every candidate is nonfinite raises an error. Set
`line_search=False` to apply `learning_rate` directly after checking the resulting loss.
`max_update_norm` optionally clips the direction in the quadratic model norm.

### Supported surface

| Supported | Notes |
|---|---|
| `FunctionalSolver` with one or more `phydrax.terms.ResidualPenalty` terms | Every training term must expose nonnegative quadratic residual roots. |
| Pointwise flat `phydrax.nn.MLP` fields | Scalar or tensor outputs; scanned execution, ordinary skip connections, and learned skip projections are supported. |
| Soft and hard-enforced PINNs | Hard ansätze remain in the differentiated residual graph. |
| Coupled fields and inverse scalar parameters | Small parameters outside affine MLP blocks use one exact dense block up to `exact_block_max_size`. |
| Mean and integral reductions | Component masks, quadrature masses, global weights, pointwise weights, and adaptive batch weights are preserved. |
| Term subsampling and adaptive collocation | Per-term factors persist across inactive steps. |

| Rejected by default | Reason or explicit alternative |
|---|---|
| Raw signed scalar terms and attached model losses | They do not provide nonnegative residual roots for the GGN model. Express a nonnegative square as a `ResidualPenalty`. |
| Non-`ResidualPenalty` training terms | No structured residual-root contract. |
| Structured, blockwise, or axis-batched models | The initial implementation covers pointwise flat `MLP` fields only. |
| Random-weight-factorized, positive-weight, or complex affine layers | Their parameterization is not an ordinary real affine block. |
| Active dropout or other stochastic model execution | Gradient, curvature, and line search would not share a deterministic function. |
| Shared/reused affine parameters | Block ownership would be ambiguous. |
| More than `exact_block_max_size` uncovered scalars | Set `uncovered="diagonal"` for an explicit diagonal fallback, or raise the exact threshold deliberately. |
| Materially negative or nonfinite residual-reduction coefficients | They do not define real square-root-weighted residual roots; negative roundoff no smaller than `-1e-12` is clamped to zero. |

The `jit` argument to `FunctionalSolver.solve` remains API-compatible. KFAC currently uses
an eager Python control loop around compiled JAX kernels; diagnostics record whether JIT
was requested.

### Configuration

- `learning_rate`: initial Armijo step, or the fixed step when line search is disabled.
- `damping`: positive isotropic regularization for every block solve.
- `factor_decay`: EMA coefficient in `[0, 1)`; the first observation initializes a factor
  exactly.
- `factor_update_period`: optimizer steps between factor refreshes.
- `factor_chunk_size`: maximum number of residual roots differentiated together for one
  parameter block.
- `approximation`: `"expand"` for all residual-event modes or `"reduce"` for the leading
  mode.
- `cg_max_steps`, `cg_relative_tolerance`: matrix-free block-solve stopping controls.
- `exact_block_max_size`: largest jointly solved non-affine parameter block.
- `uncovered`: `"error"` or the explicit `"diagonal"` fallback for a larger uncovered
  block.
- `max_update_norm`: optional quadratic-model norm bound.
- `line_search`, `line_search_shrink`, `line_search_c1`, `line_search_max_steps`: frozen-
  batch Armijo controls.

### Diagnostics

The returned solver stores KFAC metrics in `training_diagnostics`, including factor-update
count, accepted step size, line-search steps, maximum PCG iterations and relative
residual, quadratic update norm, parameter and block counts, a damped factor-condition
estimate, and—when `profile_adaptive=True`—gradient, factor, linear-solve, line-search,
first-step, and steady-step wall times. Scalar optimization metrics are emitted to
TensorBoard when enabled; `log_terms=True` additionally emits the same per-term
training/evaluation values and data/adaptive metrics as the standard solver path.

### Benchmark campaign

`tools/kfac_pinn_campaign.py` provides matched Poisson, heat, Burgers,
high-dimensional, coupled-field, and inverse-problem cases for Adam, upstream
`optax.lbfgs()` with its zoom line search, KFAC-expand, KFAC-reduce, and an explicit
exact-GGN oracle limited to tiny models.

```bash
python tools/kfac_pinn_campaign.py \
  --cases poisson-1d heat-1d burgers-1d \
  --optimizers adam lbfgs kfac-expand kfac-reduce \
  --widths 16 32 --depths 2 3 --seeds 0 1 2
```

Each configuration runs in an isolated subprocess and emits one JSON line, so compilation
caches and backend peak-memory counters do not leak across sweep points. Records include
initial/final loss, relative loss, wall time, parameter count, logical collocation
point-steps, first- and steady-step times, first-step overhead, KFAC phase timings,
factor updates, a factor condition estimate, and peak device memory when the backend
reports it. First-step overhead is omitted when periodic KFAC factor updates make the
first- and post-first-step phase schedules structurally different. Use `--smoke` for a
bounded eight-point run.

::: phydrax.optim.kfac

## Convex quadratic programs

`QuadraticProgram` represents the batched canonical convex program
`minₓ 0.5 xᵀQx + qᵀx` subject to `Ax = b` and `Gx ≤ h`. Pass `Q` and `q` as
`quadratic` and `linear`; the equality and inequality arrays are optional. A
missing constraint family becomes a zero-row array rather than a different
representation. All inputs broadcast over a common `batch_shape`, and the
primal, dual, slack, diagnostic, validity, and status arrays preserve those
batch axes.

Convexity is a caller precondition: the symmetric part of `Q` must be positive
semidefinite. The solver does not project or repair `Q`, add hidden jitter, or
replace a failed method with another one. `regularization` is the only
regularization control and is recorded in the result. There is no public
warm-start argument, elastic repair, or backend fallback.

!!! example
    ```python
    import jax.numpy as jnp
    import phydrax as phx

    problem = phx.optim.QuadraticProgram(
        jnp.asarray([[2.0, 0.0], [0.0, 2.0]]),
        jnp.asarray([-2.0, -5.0]),
        equality_matrix=jnp.asarray([[1.0, 1.0]]),
        equality_rhs=jnp.asarray([2.0]),
        inequality_matrix=jnp.asarray([[-1.0, 0.0], [0.0, -1.0]]),
        inequality_rhs=jnp.asarray([0.0, 0.0]),
    )
    result = phx.optim.solve_quadratic_program(problem)
    assert result.successful

    # Use the primal-only API when the solution belongs in a differentiated graph.
    differentiable_x = phx.optim.solve_quadratic_program_primal(
        problem,
        method="dense-active-set",
    )
    ```

### Methods, differentiation, and dense guard

`solve_quadratic_program` returns the complete audited result. Its default
`method="dense-primal-dual"` uses the Phydrax dense primal-dual implementation.
`method="qpax-implicit"` selects the required QPax 0.1.4 runtime dependency
explicitly and records that backend. If its call fails, Phydrax does not silently
run the dense solver.
The public `QPMethod` type is
`Literal["dense-primal-dual", "qpax-implicit"]`; the primal-only
`QPDifferentiableMethod` type is
`Literal["dense-active-set", "qpax-implicit"]`.

`solve_quadratic_program_primal` is the differentiable, primal-only surface.
Its default `method="dense-active-set"` custom VJP differentiates the locally
fixed active KKT system. An inequality is active only when its slack is at most
`active_set_tolerance` and its multiplier is greater than that tolerance.
These derivatives are local to that selected active set and are not
differentiable through an active-set change; an invalid forward solution
produces NaN data gradients rather than a fabricated sensitivity.

The only QPax differentiation route is `method="qpax-implicit"`, which calls
QPax's public implicit custom-VJP primal API. QPax explicit differentiation is
not accepted. The full-result and primal-only APIs share `tolerance`,
`max_iterations`, `regularization`, and `max_dense_dimension`. `step_fraction`
configures only the native dense solver. QPax 0.1.4 hard-codes a 0.99
fraction-to-boundary step and exposes no equivalent option, so QPax calls reject any
non-default public `step_fraction` request instead of silently ignoring it. The
primal-only API additionally exposes `active_set_tolerance`.

Both backends are subject to the same dense-system guard. For `n` primal
variables, `m` equalities, and `p` inequalities, the guarded dimension is
`n + m + 2p`; it must not exceed `max_dense_dimension`, whose default is 512.
This is a rejection boundary, not a request to approximate or switch methods.

### Result and KKT audit

`QuadraticProgramResult` keeps `primal`, `equality_dual`,
`inequality_dual`, and `inequality_slack` separately. For `Gx ≤ h`, both the
slack and inequality multiplier are nonnegative, the slack equation is
`Gx + slack - h = 0`, and complementarity is
`slack * inequality_dual = 0`.

The result reports the original objective and the complete audit:

- `stationarity_residual` and `dual_residual_norm` use the original `Q`;
- `solver_stationarity_residual` and `solver_dual_residual_norm` include the
  requested `regularization * primal`;
- `equality_residual`, `inequality_residual`, and
  `inequality_violation` preserve their full constraint axes;
- `complementarity_residual`, `complementarity_gap`,
  `primal_residual_norm`, and `kkt_residual_norm` expose the remaining KKT
  checks;
- `iterations` and `backend_converged` preserve the backend report, while
  Phydrax independently derives `valid` and `status` from finite inputs and
  outputs, multiplier/slack signs, feasibility, complementarity, and the
  regularized solver KKT norm;
- `method`, `backend`, `regularization`, `tolerance`, and `max_iterations`
  retain the numerical provenance.

Status is an integer array with public constants `QP_SUCCESS`,
`QP_MAX_ITERATIONS`, `QP_INFEASIBLE`, and `QP_NONFINITE`.
`result.successful` is the batch-shaped boolean success predicate. A backend
convergence flag alone is not promoted to success when the independent KKT
audit fails.

::: phydrax.optim.QuadraticProgram
    options:
        members:
            - __init__

---

::: phydrax.optim.QuadraticProgramResult
    options:
        members:
            - successful

---

::: phydrax.optim.solve_quadratic_program

---

::: phydrax.optim.solve_quadratic_program_primal

## Bounded global search: differential evolution

`DifferentialEvolutionSearch` configures a fixed-dimensional, bounded population
search. It supports `"best1bin"` and `"rand1bin"`, requires a population of at least
four, and uses a typed `phydrax.sampling` reference design for initialization. Latin
hypercube is the default; scrambled Sobol and the other supported designs are available
when their design guarantees fit the requested population size.

The root PRNG key determines initialization and every generation. The current point is
inserted as population member zero. Mutation overshoot is repeatedly reflected into the
closed box, the whole population is evaluated with `jax.vmap`, and non-finite objective
values are counted and treated as positive infinity for selection. Population
convergence means finite objective dispersion satisfies:

`standard_deviation <= absolute_tolerance + relative_tolerance * abs(mean)`.

It is not evidence that every basin was covered or that the global optimum was proved.
Exact generation, objective-evaluation, and invalid-evaluation counts are reported by the
consuming adapter.

Use the domain adapter matching the objective contract:

- `DesignConstraintSystem.search(...)` for compiled geometry design states;
- `phydrax.uq.search_map(...)` for full-data posterior densities in unconstrained
  posterior-position coordinates.

The configuration is not a generic callback-based public optimizer. New workflows should
expose explicit bounds, reconstruction, validity, and result semantics rather than
bypassing their domain contract.

::: phydrax.optim.DifferentialEvolutionSearch
    options:
        members:
            - __init__

## External optimizer compatibility

| Object | `FunctionalSolver` | `fit_operator` | Geometry/UQ search |
|---|---:|---:|---:|
| Optax transformations | Yes | Yes | No |
| `phydrax.optim.kfac` | Yes | No | No |
| `DifferentialEvolutionSearch` | No | No | Yes, through semantic adapters |
| Optimistix solvers | No direct adapter yet | No direct adapter yet | No |

`FunctionalSolver.solve` accepts standard and extra-argument Optax transformations.
Use `optax.lbfgs()` for the supported L-BFGS path; Phydrax does not maintain a parallel
quasi-Newton implementation. Operator fitting also accepts supplied Optax
transformations. Resumable `fit_operator` runs require a stable `optimizer_id` whenever
the transformation is supplied externally so checkpoint identity does not depend on an
opaque Python object.

Evosax distribution-based algorithms remain accepted by `FunctionalSolver`; its
population-based algorithms require an explicit finite search-space contract and are
rejected there. Optimistix remains an internal dependency for specific equation solves,
not a public optimization adapter.
