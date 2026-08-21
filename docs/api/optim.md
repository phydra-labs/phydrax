# Optimization

## Ownership and backend model

`phydrax.optim` is the sole namespace for optimization algorithms and configurations
owned by Phydrax. It is intentionally narrow: each object exposes a concrete numerical
contract, while the consuming workflow retains its geometry, sampling, objective,
reconstruction, validity, checkpoint, and result semantics.

Phydrax does not mirror upstream Optax, Evosax, or Optimistix APIs. Import those objects
from their native packages. A workflow accepts an external optimizer only when it has an
explicit adapter for that optimizer family.

## Finite exhaustive search

`FiniteAxis` and `FiniteProductSpace` represent an explicit, array-backed finite
candidate set. A `FiniteAxis` stores one correlated catalog: every array leaf has the
same nonempty leading candidate dimension, while all trailing dimensions remain one
candidate payload. Use separate axes only for choices that should form a Cartesian
product.

!!! example
    ```python
    import jax.numpy as jnp
    import phydrax as phx


    # One correlated catalog of complete two-coefficient choices.
    coefficient_catalog = phx.optim.FiniteProductSpace(
        phx.optim.FiniteAxis(
            jnp.asarray(
                [
                    [0.0, 0.0],
                    [0.5, 0.5],
                    [1.0, 0.0],
                ]
            )
        )
    )

    # Two independent coordinate catalogs; this space has 3 × 2 candidates.
    coordinate_product = phx.optim.FiniteProductSpace(
        {
            "offset": phx.optim.FiniteAxis(jnp.asarray([-1.0, 0.0, 1.0])),
            "slope": phx.optim.FiniteAxis(jnp.asarray([0.0, 2.0])),
        }
    )
    assert coordinate_product.product_shape == (3, 2)
    assert coordinate_product.size == 6
    ```

The product is lazy: Phydrax stores the axis arrays, not a materialized Cartesian
candidate tensor. Candidate order follows deterministic JAX PyTree path order and the
last axis varies fastest. `ravel_index`, `unravel_index`, and `take` use this same
row-major convention. Checked public indexing rejects negative and oversized indices;
it never clips them. `signature()` hashes axis paths, grouping, shapes, dtypes, and
candidate bytes for content-sensitive provenance.

`FiniteExhaustiveSearch` configures exact enumeration for domain adapters such as
`phydrax.uq.search_map_candidates` and
`phydrax.control.search_control_candidates`. `batch_size=None` uses scalar streaming.
A positive batch size evaluates complete batches plus one exact-size remainder, so no
candidate is padded, repeated, or omitted. Every declared candidate is evaluated
exactly once. Invalid or nonfinite values are excluded before reduction, and equal
finite values select the lowest flat index deterministically.

The guarantee is the minimum over the declared finite set. It is not a certificate of
continuous global optimality, an integration method, or a marginalization rule.
Selection is discrete and gradients are stopped through the selected index and
reconstructed candidate. The current execution kernel is single-device; it makes no
GPU, TPU, or distributed-sharding scaling claim.

Factorized storage avoids materializing a candidate landscape, but it cannot remove
Cartesian growth. If axis lengths are `n₁, …, nₖ`, the evaluator still runs
`n₁ × ⋯ × nₖ` times. Batch size trades transient candidate/output storage for
throughput. Candidate payloads remain correlated within an axis; split them only when
the full cross product is intended.

Run the factorized-streaming and bounded dense-oracle benchmark with:

```bash
python tools/finite_search_benchmarks.py \
    --axes 3 --axis-length 16 --batch-size 64
```

The JSON report records candidate cardinality, stored and estimated dense bytes,
compilation and steady execution time, compiler-reported memory, exact evaluation
counts, and selected value/index agreement. Dense materialization is skipped when its
estimate exceeds `--max-dense-bytes`.

The array-backed lazy-product pattern was independently implemented after reviewing
[Brutax](https://github.com/michael-0brien/brutax) as design inspiration. Phydrax does
not depend on Brutax; its reducer, validity, evidence, provenance, domain adapters, and
result contracts are Phydrax-owned.

::: phydrax.optim.FiniteAxis

---

::: phydrax.optim.FiniteProductSpace

---

::: phydrax.optim.FiniteExhaustiveSearch

## Riemannian optimization

`ParameterGeometry` binds complete trainable PyTree leaves to explicit
`AbstractRiemannianManifold` instances and can assign each selected leaf a positive
static product-metric weight. Fixed-step SGD, transported momentum, intrinsic
Adam/AMSGrad, frozen-objective Armijo search, Riemannian conjugate gradient, and
Riemannian L-BFGS all share this geometry. Unselected inexact-array leaves retain
ordinary Euclidean geometry.

!!! example
    ```python
    import phydrax as phx


    def train_riemannian(solver):
        parameters = solver.trainable_functions()
        paths = phx.optim.ParameterGeometry.array_leaf_paths(parameters)
        orthogonal_path = next(path for path in paths if "weight" in path)

        geometry = phx.optim.ParameterGeometry.from_leaf_paths(
            parameters,
            {
                orthogonal_path: phx.metrix.StiefelManifold(32, 8),
            },
        )
        optimizer = phx.optim.riemannian_sgd(
            geometry,
            learning_rate=1e-2,
            max_gradient_norm=1.0,
        )
        return solver.solve(
            num_iter=1_000,
            optim=optimizer,
            keep_best=False,
        )
    ```

### Parameter PyTree binding

Paths use the deterministic `jax.tree_util.keystr` representation returned by
`ParameterGeometry.array_leaf_paths`. A binding records the complete trainable tree
definition, every leaf shape, and every dtype. Reusing it with a different structure
fails before the objective is evaluated.

Selected leaves must:

- be real floating-point JAX arrays;
- have the manifold's `point_shape` as their trailing shape;
- satisfy manifold membership at construction.

Leading axes are a product of independent points. Selection currently applies to a
whole trainable leaf, not a slice inside a leaf. Phydrax does not infer Stiefel, SO,
SPD, or another geometry from shape or initial value.

The product metric is the sum of declared leaf metrics. Global gradient clipping uses
one scale for this norm; it does not clip each leaf independently.

Adaptive moments follow the same product decomposition. A selected leaf stores one
scalar second moment per independent leading-axis point, never one value per ambient
matrix or vector coordinate. An unselected Euclidean leaf is one whole factor and
therefore has one scalar moment. To request ordinary pointwise Euclidean Adam explicitly,
bind that leaf to `EuclideanManifold(())`; its array axes then become leading product
axes, with one scalar factor per element.

### Algorithms

`riemannian_sgd` performs metric gradient conversion, optional global clipping, and one
retraction per selected leaf. On Euclidean leaves it is exactly ordinary fixed-step
gradient descent.

`riemannian_momentum` implements heavy-ball momentum. The momentum is tangent at the
current point, determines the retraction step, and is transported to the destination
tangent space after every update. It is not ambient momentum followed by projection
only at output.

`riemannian_adam` transports its full tangent first moment after every retraction. Its
nonnegative second moment is the declared factor's weighted metric squared norm, with
standard exponential decay and bias correction. The resulting denominator scales the
whole tangent factor uniformly, so Stiefel columns, SPD matrices, and other represented
manifold points are never divided coordinatewise. Set `amsgrad=True` to use the running
maximum of each factor moment.

`riemannian_conjugate_gradient` implements Polak--Ribière+ directions. Previous
gradients and directions are transported to every accepted point; the method restarts
when conjugacy no longer defines descent.

`riemannian_lbfgs` stores transported tangent-space secant pairs in bounded,
static-shape history. Pairs with nonfinite or insufficient curvature are rejected.
Both line-search methods evaluate one frozen objective closure through Armijo
backtracking and return search counts and reduction diagnostics in optimizer state.

The fixed-step optimizers accept a positive scalar learning rate or a callable schedule
receiving the zero-based JAX step scalar. A scheduled value may be zero, but must remain
finite and nonnegative. Nonfinite metric gradient norms fail explicitly. Riemannian Adam
also accepts first- and second-moment decays in `[0, 1)`, a positive denominator
`epsilon`, and optional AMSGrad.

Arbitrary Optax transform composition is intentionally unsupported. Ambient
coordinatewise moments on a non-Euclidean factor, ambient weight decay, and momentum
without transport are not invariant under changes of representation.
`evaluation_parameters` is also rejected because an ambient evaluation transform need
not preserve membership.

### Supported surface

| Supported | Notes |
|---|---|
| Mixed trainable PyTrees | Selected leaves use manifold updates; all other inexact leaves are Euclidean. |
| Sphere, hyperbolic, simplex, Stiefel, Grassmann, oblique, fixed-rank, SO(n), affine-invariant SPD(n) | See [Array manifolds](metrix/manifolds.md) for exact metric and transport semantics. |
| Leading products/batches | Geometry dimensions are trailing; all leading axes are preserved. |
| `FunctionalSolver` terms and model losses | Existing term evaluation, sampling, adaptive collocation, enforcement, and best-parameter logic are reused. |
| Eager and JIT execution | SGD, momentum, Adam/AMSGrad, and the line-search update kernels are JAX-transformable. |

| Rejected | Reason |
|---|---|
| Automatic manifold discovery | Shapes do not determine intended geometry or metric. |
| Selected complex leaves | The current built-ins have real manifold contracts. |
| Subarray selection | It would introduce overlapping ownership and scatter semantics. Split the parameter into explicit leaves. |
| Arbitrary Optax chaining | Generic transform ordering does not preserve tangent-state semantics. |
| Manifold KFAC | No invariant curvature-factor contract is currently implemented. |

### Diagnostics and benchmark

Console and TensorBoard output include Riemannian gradient, tangent-step, and momentum
norms; clipping, constraint, tangent, and transport residuals; adaptive denominator
bounds; and Armijo evaluation, acceptance, reduction, conjugacy, and active-history
diagnostics. Returned `training_diagnostics` additionally records the number of selected
manifold leaves.

Run the invariant-aware smoke benchmark with:

```bash
python -m tools.riemannian_optim_benchmarks --smoke
```

It reports compile-plus-first and steady update times, explicit transport timings,
output bytes, objective progress, Armijo behavior, and sphere, Stiefel, SO, SPD, and
mixed-PyTree constraint diagnostics without comparisons to external libraries.

::: phydrax.optim.ParameterGeometry
    options:
        members:
            - from_leaf_paths
            - array_leaf_paths
            - validate
            - contains
            - constraint_residuals
            - maximum_constraint_residual

---

::: phydrax.optim.riemannian_sgd

---

::: phydrax.optim.riemannian_momentum

---

::: phydrax.optim.riemannian_adam

---

::: phydrax.optim.ArmijoLineSearch

---

::: phydrax.optim.riemannian_conjugate_gradient

---

::: phydrax.optim.riemannian_lbfgs

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
| Pointwise flat `phydrax.nn.models.MLP` fields | Scalar or tensor outputs; scanned execution, ordinary skip connections, and learned skip projections are supported. |
| Soft and hard-enforced PINNs | Hard ansätze remain in the differentiated residual graph. |
| Coupled fields and inverse scalar parameters | Small parameters outside affine MLP blocks use one exact dense block up to `exact_block_max_size`. |
| Mean and integral reductions | Component masks, quadrature masses, global weights, pointwise weights, and adaptive batch weights are preserved. |
| Term subsampling and adaptive collocation | Per-term factors persist across inactive steps. |

| Rejected by default | Reason or explicit alternative |
|---|---|
| Raw signed scalar terms and attached model losses | They do not provide nonnegative residual roots for KFAC's structured GGN blocks. Use `GeneralizedGaussNewton` for a mixed residual-plus-scalar solve, or express a genuinely nonnegative square as a `ResidualPenalty`. |
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

## Typed nonlinear optimization

Phydrax exposes one status, diagnostics, and provenance model across its native scalar,
least-squares, constrained, state/design, and stochastic optimization algorithms.
`MinimizationProblem` owns a scalar objective, optional `Bounds`, optional
`NonlinearConstraint` objects, and a stable problem identifier. An objective returns
either a scalar value or `(value, auxiliary)`. `NonlinearLeastSquaresProblem` instead
owns a real residual PyTree and defines its objective as one half of the squared residual
norm.

`OptimizationTermination` separates absolute and relative optimality, step, feasibility,
maximum-step, and evaluation-budget controls. `maximum_evaluations` is checked between
nonlinear iterations: one indivisible globalization step and the final result-packaging
evaluation may increase the reported counter beyond that gate. Large static step budgets
do not unroll Python; native runtimes stage their loops with JAX control flow.

Methods return `MinimizationResult` with accepted parameters, the corresponding objective
and auxiliary output, a typed `OptimizationStatus`, numerical
`OptimizationDiagnostics`, and `OptimizationProvenance`. Success is explicit; reaching a
step or evaluation limit, encountering nonfinite data, failing a linear solve, or failing
restoration never becomes a nominally successful result. Trial points never replace the
accepted point unless globalization accepts them.
Globalization distinguishes a finite trial that fails sufficient-decrease or
trust-region acceptance from a search whose every evaluated trial is nonfinite; the
latter returns `NONFINITE_EVALUATION` while preserving the last accepted point.

### Unconstrained scalar, residual, and composite methods

`minimize` dispatches only through a declared Phydrax method. `NewtonKrylov` uses
matrix-free Hessian-vector products, the shared linear-solve policy layer,
Eisenstat--Walker style inexact forcing, and frozen-objective Armijo globalization.
Indefinite or unusable Newton directions fall back to steepest descent and record that
decision. `NewtonTrustRegion` instead solves a damped Newton model and updates its
radius from the actual-to-predicted reduction ratio. `NonlinearConjugateGradient`
supports Fletcher--Reeves, Polak--Ribière+, Hestenes--Stiefel+, and Dai--Yuan beta
rules, with periodic, orthogonality, and lost-descent restarts. Its
`StrongWolfeLineSearch` reports the Armijo and curvature inequalities independently.

`OptimistixMethod` is the explicit interoperability boundary for an upstream
Optimistix minimizer; it preserves backend ownership of tolerance tests while
normalizing the public result and provenance. Neither unconstrained route silently
drops bounds or nonlinear constraints. Optimistix does not expose portable
objective-evaluation counts. Adapter diagnostics therefore set
`counts_complete=False`, and `maximum_evaluations` is rejected rather than silently
approximated.

`least_squares` accepts `GaussNewton`, `LevenbergMarquardt`, and
`FiniteDifferenceGaussNewton`. The differentiated methods use matrix-free Jacobian
products over arbitrary real array PyTrees. Levenberg--Marquardt accepts or rejects
trial steps by the actual-to-predicted reduction ratio and updates damping accordingly;
rejected steps do not mutate the accepted iterate. The finite-difference method
constructs a deterministic coordinate Jacobian, accounts for every residual evaluation,
and does not advertise implicit differentiation.

!!! example
    ```python
    import jax.numpy as jnp
    import phydrax as phx


    problem = phx.optim.NonlinearLeastSquaresProblem(
        lambda x, _: jnp.asarray([x[0] ** 2 - 2.0]),
        problem_id="square-root",
    )
    result = phx.optim.least_squares(
        problem,
        jnp.asarray([1.0]),
        method=phx.optim.LevenbergMarquardt(),
        termination=phx.optim.OptimizationTermination(
            absolute_optimality=1e-10,
            maximum_steps=50,
        ),
    )
    assert result.successful
    ```

### Composite proximal methods

`ProximalProblem` separates a differentiable scalar objective from one
`AbstractProximalFunctional`. `ProximalGradient`, accelerated proximal gradient, and
`ProximalNewton` all terminate on the composite gradient-mapping norm and return it as
`composite_stationarity`; they do not substitute the smooth gradient norm.

Built-in functionals cover L1, elastic net, box and callable indicators, simplex
projection, grouped L1/L2 penalties, and the nuclear norm. Their proximal maps act over
the complete declared PyTree leaves. Nuclear shrinkage uses singular values, group
shrinkage uses the declared axis, and indicator values are exactly zero or infinity.
The accelerated method restarts when its extrapolation ceases to be useful.

`FunctionalSolver.solve` accepts `GaussNewton`, `LevenbergMarquardt`, and
`GeneralizedGaussNewton`. It freezes the sampled objective realization for the complete
nonlinear solve and builds a residual vector from the same weighted residual roots used
by KFAC. Plain least-squares methods require residual-based terms and reject model-level
scalar losses. `GeneralizedGaussNewton` additionally retains signed
`IntegralFunctional` terms and model-level scalar losses as raw scalars: its curvature is
the residual Gauss--Newton operator plus the exact scalar Hessian action, never the square
root of an already reduced scalar objective.

### Bounds and nonlinear constraints

`Bounds` materializes scalar or PyTree lower and upper values against the parameter tree,
broadcasts each bound to its corresponding leaf, and rejects incompatible or inverted
bounds. `ProjectedGradient`, `ProjectedLBFGS`, and `ActiveSetNewton` all preserve the
closed box at every accepted iterate. Projected L-BFGS stores curvature only from
accepted projected steps. Active-set Newton solves over free variables and reports
active-set size, primal feasibility, complementarity, and direction fallbacks.

`NonlinearConstraint` represents any differentiable residual bounded componentwise by
`lower <= c(x) <= upper`; equal finite bounds are equalities. `AugmentedLagrangian`
normalizes equality and inequality multipliers, increases its penalty only when
feasibility progress is insufficient, and delegates each frozen subproblem to an
unconstrained inner method. `SQP` forms explicit local QP subproblems through
`solve_quadratic_program`. Its optional filter keeps objective and violation dominance
separate; its second-order correction relinearizes constraints at the trial point to
address the Maratos effect. Elastic restoration has a distinct failure status.
`PrimalDualPredictorCorrector` performs an affine predictor and centered corrector,
keeps primal, dual, slack, and complementarity residuals separate, and explicitly
rejects infeasible or nonfinite initial states. Nested methods aggregate only observable
work counters; unavailable counts remain explicitly incomplete.

!!! example
    ```python
    constraint = phx.optim.NonlinearConstraint(
        lambda x, _: jnp.asarray([x[0] + x[1]]),
        lower=1.0,
        upper=1.0,
    )
    constrained = phx.optim.MinimizationProblem(
        lambda x, _: jnp.sum(x**2),
        constraints=(constraint,),
        problem_id="minimum-norm-sum",
    )
    result = phx.optim.minimize(
        constrained,
        jnp.asarray([0.0, 0.0]),
        method=phx.optim.SQP(),
    )
    ```

### Implicit solution maps

`implicit_minimize` and `implicit_least_squares` expose converged unconstrained
solutions as differentiable functions of `args`. Their derivatives solve the
stationarity or normal-equation tangent system rather than differentiating through
optimizer iterations. The initial guess selects the local solution branch and has zero
implicit sensitivity.

`implicit_constrained_minimize` applies the same contract to bounds, nonlinear
inequalities, and equalities. It canonicalizes every finite lower and upper side, solves
the primal problem through any method declaring `implicit_differentiation`, and
differentiates the locally fixed active-set KKT system with the configured
`LinearSolvePolicy`. Native projected methods, `AugmentedLagrangian`, `SQP`, and
`PrimalDualNewtonKrylov` return the multiplier, slack, activity, and stationarity
certificate required by this path.

The constrained derivative is classical only at a successful regular KKT point with a
strictly complementary active set. A failed primal solve, ambiguous activity, or
singular active KKT matrix raises an explicit runtime error; Phydrax does not choose a
subgradient at an active-set transition. The solution map supports forward and reverse
mode, filtered JIT, nested parameter and argument PyTrees, and batching. Constraint
topology and bound roles remain static across transformed calls.

!!! example
    ```python
    import jax
    import jax.numpy as jnp
    import phydrax as phx


    problem = phx.optim.MinimizationProblem(
        lambda x, target: 0.5 * jnp.sum((x - target) ** 2),
        bounds=phx.optim.Bounds(-jnp.inf, 1.0),
    )

    def solution(target):
        return phx.optim.implicit_constrained_minimize(
            problem,
            jnp.asarray([0.0]),
            args=target,
        )[0]

    active_sensitivity = jax.grad(solution)(jnp.asarray(2.0))
    assert jnp.allclose(active_sensitivity, 0.0)
    ```


### State/design optimization

`StateDesignProblem` keeps the state equation, design objective, and design bounds
separate. `ReducedAdjoint` solves the state to convergence for each accepted design,
computes the reduced gradient with one transpose linear solve, and uses the bound solver
contract for the outer step. `SimultaneousKKT` instead solves the coupled state,
adjoint, and stationarity system. Both return `StateDesignResult`, retaining state and
design separately plus state-residual, optimality, feasibility, and linear-solve
diagnostics. No method silently differentiates through an unconverged state solve.

### Stochastic objectives and decomposition

`StochasticProblem` combines a scenario loss, an explicit `AbstractSamplingPolicy`, a
law-invariant risk, optional bounds, and optional chance constraints. `FixedSampling` and
`MonteCarloSampling` make frozen versus per-iteration refresh semantics explicit.
`ExpectationRisk`, `MeanVarianceRisk`, `CVaRRisk`, and `EntropicRisk` operate on
normalized nonnegative scenario weights.

`StochasticAdam` is the stochastic-gradient baseline. `ProgressiveHedging` and
`ConsensusADMM` own scenario-local variables, consensus updates, and duals, and report
primal and dual consensus residuals. `ChanceConstraint` exposes both empirical
feasibility and a differentiable smooth estimate. The smooth estimate is converted into
a `NonlinearConstraint` only through `StochasticProblem.frozen`; stochastic-gradient
methods reject chance constraints rather than silently adding an undocumented penalty.

`ProgressiveHedging` and `ConsensusADMM` deliberately keep scenario orchestration on
the host because their public contract accepts an arbitrary typed inner optimization
method and retains each inner result's provenance. Their fixed-shape
consensus/dual/residual algebra is compiled; the outer arbitrary-method workflow is not
advertised as one whole-program `jit`/`vmap` kernel.
If an inner method reports incomplete counts, consensus workflows may still run without
an evaluation budget, but they never add negative sentinel counts and reject
`maximum_evaluations` because that budget cannot be enforced truthfully.

### End-to-end workflow examples

#### Residual plus signed scalar

`CompositeLeastSquaresProblem` represents the objective
`0.5 * ||residual(parameters)||**2 + scalar(parameters)` without forcing the scalar term
into a residual square:

```python
import jax.numpy as jnp
import phydrax as phx


problem = phx.optim.CompositeLeastSquaresProblem(
    lambda x, _: x - 1.0,
    lambda x, _: -0.1 * x[0],
)
result = phx.optim.composite_least_squares(
    problem,
    jnp.zeros(1),
    method=phx.optim.GeneralizedGaussNewton(),
)
assert result.successful
assert jnp.allclose(result.parameters, jnp.array([1.1]), atol=1e-7)
```

#### Differentiable bilevel solve

Use an implicit solution map inside the outer objective. Its reverse-mode derivative
solves the stationarity tangent system rather than storing the inner iterations:

```python
import jax
import jax.numpy as jnp
import phydrax as phx


def validation_loss(training_target):
    fitted = phx.optim.implicit_minimize(
        lambda weight, target: (
            0.5 * jnp.sum((weight - target) ** 2)
            + 0.1 * jnp.sum(weight**2)
        ),
        jnp.zeros(1),
        args=training_target,
    )
    return jnp.sum((fitted - 1.5) ** 2)


hypergradient = jax.grad(validation_loss)(jnp.array([2.0]))
```

#### Constrained solve

```python
import jax.numpy as jnp
import phydrax as phx


constraint = phx.optim.NonlinearConstraint(
    lambda x, _: jnp.array([x[0] + x[1]]),
    lower=1.0,
    upper=1.0,
)
problem = phx.optim.MinimizationProblem(
    lambda x, _: jnp.sum(x**2),
    bounds=phx.optim.Bounds(0.0, 1.0),
    constraints=(constraint,),
)
result = phx.optim.minimize(
    problem,
    jnp.array([0.2, 0.8]),
    method=phx.optim.SQP(),
)
assert result.successful
assert result.diagnostics.primal_feasibility < 1e-7
```


### Runtime and benchmark evidence

Native scalar, residual, composite, bounded, nonlinear-constrained, state/design, and
stochastic-gradient numerical loops are JAX-staged. Explicit iterations support eager
execution, filtered JIT, forward-mode JVPs, dynamic arguments, `vmap`, large maximum-step
budgets, and partitioned Equinox parameter PyTrees. Reverse-mode solution sensitivities
use `implicit_minimize`, `implicit_least_squares`, or
`implicit_constrained_minimize`; they do not reverse an unbounded dynamic iteration tape.

Curvature methods prepare one symbolic linear-solve template, then refresh only numeric
state. `setup_refreshes` counts template preparations; `numeric_refreshes` counts every
numeric bind and rebind of that template, including the initial bind. The
`prepared_refresh` capability is true only for methods using this lifecycle.
Refresh state exposes only array leaves to JAX; immutable template and callable
preconditioner structure is held as explicit static Equinox metadata rather than a
dynamic loop carry.

Run the benchmark suite with:

```bash
python -m tools.nonlinear_optimization_benchmarks
```

Use `--smoke` for a bounded contract check. JSON records report eager execution, separate
JIT compilation, first and steady compiled execution, first versus steady optimizer
steps, prepared setup/refresh counts, matrix-free scaling at multiple sizes, dense-memory
counterfactuals, frozen-objective line-search accounting, constrained and state/design
solves, and stochastic common-random-number behavior. Timings are descriptive
single-process measurements, not backend-independent performance claims.

::: phydrax.optim.MinimizationProblem

---

::: phydrax.optim.NonlinearLeastSquaresProblem

---

::: phydrax.optim.OptimizationTermination

---

::: phydrax.optim.MinimizationResult

---

::: phydrax.optim.OptimizationCapabilities

---

::: phydrax.optim.OptimizationProvenance

---

::: phydrax.optim.ConstrainedOptimalityCertificate

---

::: phydrax.optim.NewtonKrylov

---

::: phydrax.optim.NewtonTrustRegion

---

::: phydrax.optim.NonlinearConjugateGradient

---

::: phydrax.optim.StrongWolfeLineSearch

---

::: phydrax.optim.OptimistixMethod

---

::: phydrax.optim.GaussNewton

---

::: phydrax.optim.LevenbergMarquardt

---

::: phydrax.optim.FiniteDifferenceGaussNewton

---

::: phydrax.optim.ProximalProblem

---

::: phydrax.optim.ProximalGradient

---

::: phydrax.optim.AcceleratedProximalGradient

---

::: phydrax.optim.ProximalNewton

---

::: phydrax.optim.CompositeLeastSquaresProblem

---

::: phydrax.optim.CompositeLeastSquaresResult

---

::: phydrax.optim.GeneralizedGaussNewton

---

::: phydrax.optim.composite_least_squares

---

::: phydrax.optim.Bounds

---

::: phydrax.optim.NonlinearConstraint

---

::: phydrax.optim.ProjectedGradient

---

::: phydrax.optim.ProjectedLBFGS

---

::: phydrax.optim.ActiveSetNewton

---

::: phydrax.optim.AugmentedLagrangian

---

::: phydrax.optim.SQP

---

::: phydrax.optim.FilterGlobalization

---

::: phydrax.optim.PrimalDualNewtonKrylov

---

::: phydrax.optim.PrimalDualPredictorCorrector

---

::: phydrax.optim.implicit_minimize

---

::: phydrax.optim.implicit_least_squares

---

::: phydrax.optim.implicit_constrained_minimize

---

::: phydrax.optim.StateDesignProblem

---

::: phydrax.optim.StochasticProblem

---


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

| Object | Standalone nonlinear API | `FunctionalSolver` | `fit_operator` | Geometry/UQ search |
|---|---:|---:|---:|---:|
| Optax transformations | No | Yes | Yes | No |
| `phydrax.optim.kfac` | No | Yes | No | No |
| Phydrax scalar, least-squares, and composite methods | Yes | Yes | No | No |
| Phydrax constrained/state-design/stochastic methods | Yes | No | No | No |
| Phydrax Riemannian optimizers | No | Yes | No | No |
| `DifferentialEvolutionSearch` | No | No | No | Yes, through semantic adapters |
| `OptimistixMethod` | Yes | No | No | No |

`FunctionalSolver.solve` accepts standard and extra-argument Optax transformations,
Phydrax KFAC and Riemannian optimizers, and native scalar, least-squares, and composite
iterative methods. Operator fitting accepts supplied Optax transformations.
Resumable `fit_operator` runs require a stable `optimizer_id` whenever the transformation
is supplied externally so checkpoint identity does not depend on an opaque Python object.

Evosax distribution-based algorithms remain accepted by `FunctionalSolver`; its
population-based algorithms require an explicit finite search-space contract and are
rejected there. Optimistix interoperation is deliberately explicit and standalone:
wrap a compatible upstream minimizer in `OptimistixMethod`. Phydrax does not inspect an
arbitrary upstream object or silently reinterpret its stopping rules.
