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
zero but its parameter Jacobian is not. Constraint weights, pointwise weights, masks,
quadrature masses, and mean/integral normalization are included exactly once when the
residual roots are formed.

Each supported affine weight-and-bias block is approximated independently. Contributions
from distinct `FunctionalConstraint` terms retain separate exponential-moving-average
state and are combined as a sum of Kronecker products; no artificial cross-constraint
terms are introduced. `approximation="expand"` retains every singular mode of each
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

Every optimizer step materializes one batch for each active constraint. The same batches,
adaptive-collocation weights, evaluation keys, and iteration value are reused for the
gradient, factor update, and all Armijo candidates. Adaptive collocation is refreshed at
most once before the step. When `train_constraint_sample_size` is set, the first step
warms all per-constraint factors; later steps update only the sampled terms while
retaining their stored factors.

`line_search=True` uses frozen-batch Armijo backtracking. A failed finite search returns a
zero step; a search whose every candidate is nonfinite raises an error. Set
`line_search=False` to apply `learning_rate` directly after checking the resulting loss.
`max_update_norm` optionally clips the direction in the quadratic model norm.

### Supported surface

| Supported | Notes |
|---|---|
| `FunctionalSolver` with one or more `FunctionalConstraint` terms | Every active constraint must expose nonnegative quadratic residual roots. |
| Pointwise flat `phydrax.nn.MLP` fields | Scalar or tensor outputs; scanned execution, ordinary skip connections, and learned skip projections are supported. |
| Soft and hard-enforced PINNs | Hard ansätze remain in the differentiated residual graph. |
| Coupled fields and inverse scalar parameters | Small parameters outside affine MLP blocks use one exact dense block up to `exact_block_max_size`. |
| Mean and integral reductions | Component masks, quadrature masses, global weights, pointwise weights, and adaptive batch weights are preserved. |
| Constraint subsampling and adaptive collocation | Per-constraint factors persist across inactive steps. |

| Rejected by default | Reason or explicit alternative |
|---|---|
| Standalone/signed `objectives` and attached model losses | They do not provide nonnegative residual roots for the GGN model. Express a nonnegative square as a `FunctionalConstraint`. |
| Non-`FunctionalConstraint` terms | No structured residual-root contract. |
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
TensorBoard when enabled; `log_constraints=True` additionally emits the same per-term
training/evaluation losses and data/adaptive metrics as the standard solver path.

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
