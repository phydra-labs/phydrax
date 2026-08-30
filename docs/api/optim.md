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

## Mirror descent

`ParameterMirrorGeometry` binds selected real PyTree leaves to explicit
`LegendreGeometry` instances. Unselected leaves use the Euclidean quadratic potential
and ordinary additive gradient descent. The binding records the complete tree
definition, every leaf shape, and every dtype; it never infers a mirror map from a
positive initial value.

For a selected potential `Φ` and coordinate gradient `g`, `mirror_descent` applies

`x⁺ = (∇Φ)⁻¹(∇Φ(x) - αg/w)`,

where `w > 0` is the optional static geometry weight. This is a direct dual-coordinate
translation. It does not calculate `∇²Φ`, solve a natural-gradient system, or project
an invalid result back into the domain.

!!! example
    ```python
    import phydrax as phx


    def train_mirror(solver, negative_entropy_geometry):
        parameters = solver.trainable_functions()
        paths = phx.optim.ParameterMirrorGeometry.array_leaf_paths(parameters)
        positive_path = next(path for path in paths if "positive" in path)

        geometry = phx.optim.ParameterMirrorGeometry.from_leaf_paths(
            parameters,
            {positive_path: negative_entropy_geometry},
        )
        optimizer = phx.optim.mirror_descent(
            geometry,
            learning_rate=1e-2,
        )
        return solver.solve(
            num_iter=1_000,
            optim=optimizer,
            keep_best=False,
        )
    ```

See [Hessian and Legendre information geometry](metrix/information_geometry.md) for a
complete negative-entropy construction. Its update is multiplicative:

`x⁺ = x · exp(-αg/w)`.

### Binding and product semantics

A selected leaf must:

- use real floating-point coordinates;
- end in the Legendre chart dimension;
- lie strictly inside the declared primal support at construction.

Leading axes are independent copies of the same potential. A potential may couple the
coordinates within one final-axis point, but it does not couple different leading
factors or different PyTree leaves. Split a genuinely coupled parameterization into one
explicit leaf and declare one geometry for that complete coordinate vector.

Scaling a leaf geometry by `w` scales both its potential and reported step divergence:

`D_{wΦ}(x⁺ ∥ x) = w DΦ(x⁺ ∥ x)`.

The corresponding dual displacement is `-αg/w`. Weights must be finite and strictly
positive.

### Failure and diagnostics

The initial fixed-step implementation accepts a positive scalar learning rate or a
callable schedule receiving the zero-based JAX step. A scheduled rate may be zero but
must remain finite and nonnegative. Momentum, clipping, adaptive moments, and
backtracking are deliberately absent: each requires a separate dual-coordinate
contract rather than ambient Optax composition.

If a translated dual point or its inverse leaves its declared support, execution fails.
There is no clipping, normalization, jitter, or hidden step reduction.
`evaluation_parameters` is rejected because an ambient transform need not preserve
Legendre support.

Console, TensorBoard, and returned diagnostics use the `optimizer/mirror/` namespace:

- coordinate-gradient norm;
- applied dual-displacement norm;
- Bregman step divergence;
- support residual;
- selected Legendre-leaf count.

The two norms are coordinate diagnostics, not invariant natural-gradient norms.

### Relation to simplex optimization

`ProbabilitySimplexManifold` plus `riemannian_sgd` already performs exact
negative-entropy mirror descent on the open normalized simplex. Its Fisher sharp map
and multiplicative retraction simplify to

`p⁺ = normalize(p · exp(-αg))`.

Use that existing manifold path for normalized probabilities. Use `SimplexIndicator`
when boundary solutions with exact zeros are required. Do not represent the affine
mass constraint as an unconstrained full-dimensional `LegendreGeometry`.

::: phydrax.optim.ParameterMirrorGeometry
    options:
        members:
            - from_leaf_paths
            - array_leaf_paths
            - validate
            - contains
            - constraint_residuals
            - maximum_constraint_residual

---

::: phydrax.optim.mirror_descent

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
decision. `NewtonTrustRegion` builds a matrix-free Hessian action and solves its
self-adjoint quadratic model with `SteihaugToint`; negative curvature and trust-boundary
termination are explicit evidence. It has no dense dimension cap. `DenseNewtonDogleg`
retains the former bounded-dimension dense eigendecomposition and dogleg route as an
explicit small-system method. `NonlinearConjugateGradient`
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

`least_squares` accepts `GaussNewton`, `LevenbergMarquardt`,
`FiniteDifferenceGaussNewton`, `BoundedGaussNewton`, and
`BoundedLevenbergMarquardt`. The differentiated methods use matrix-free Jacobian
products over arbitrary real array PyTrees. Levenberg--Marquardt accepts or rejects
trial steps by the actual-to-predicted reduction ratio and updates damping accordingly;
rejected steps do not mutate the accepted iterate. The finite-difference method
constructs a deterministic coordinate Jacobian, accounts for every residual evaluation,
and does not advertise implicit differentiation.

Bounds on `NonlinearLeastSquaresProblem` are never converted to residual penalties.
Unbounded methods reject them. The bounded methods mask the active set, solve a
matrix-free free-variable Gauss--Newton trust model, project every candidate before
residual evaluation, and recompute predicted reduction for the actual projected
displacement.

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

### Block residual graphs and robust least squares

`ResidualGraphProblem` lowers explicit `ParameterBlock` and `ResidualBlock`
objects to the ordinary `NonlinearLeastSquaresProblem` contract. Blocks declare
parameter dependencies, weights/covariances, robust losses, bounds, manifold
retractions, and elimination groups. A one-callback residual lowers to one
parameter block and one residual block rather than creating a second convention.

Robust losses provide `rho`, first derivative, second derivative, and convex
model evidence. Native choices are identity, Huber, soft-L1, Cauchy, arctangent,
Tukey, and scaled composition. `linearize_residual_graph` assembles each factor
only over the parameter blocks it references. Its robust normal model retains
the radial second-order block curvature and clips only a negative radial
eigenvalue, with the number of clipped blocks retained as evidence.

`prepare_residual_graph` fixes block shapes, factor adjacency, and elimination
ordering. `plan_least_squares_route` selects dense QR/SVD, portable LSMR, or
Schur execution from active dimension, rank, density, and elimination evidence;
constant parameter blocks do not consume solve coordinates. `solve_residual_graph`
executes that plan, reports the route and plan IDs, and uses
`prepare_schur_plan`/`solve_schur_system` when Schur elimination is selected.
Route planning is therefore executable policy rather than advisory metadata.

The trust-region residual family includes traditional and subspace dogleg,
dogbox, trust-region reflective bounds, matrix-free bounded GN/LM, and POUNDERS
residual interpolation. `VariableProjectionProblem` eliminates a declared
linear coefficient block at every nonlinear point and differentiates the
reduced residual. `POUNDERS` is model-based derivative-free least squares; its
public success is independently recertified from finite-difference physical
stationarity and bounds, not from the interpolation model's stopping flag. It
is distinct from coordinate finite-difference Gauss--Newton.

Model-based and dense least-squares methods reuse `NonlinearPrecisionPolicy` for
parameter/model storage, direction arithmetic, widened reductions, certificate
decisions, and returned parameters. Interpolation fits, dogleg systems, POUNDERS,
variable projection, finite-difference Gauss--Newton, and Schur complements route
dense algebra through `phydrax.linalg`. `MinimizationResult` and
`LeastSquaresResult` retain the parent precision envelope; interpolation-model and
physical-certificate envelopes remain named children.

Factor graph lifecycle functions track changed factors, affected parameters,
relinearization thresholds, factor/parameter versions, and topology-changing
add/remove operations. `FactorGraphCertificate` checks objective stationarity,
finiteness, and manifold membership at the same returned parameters.

### Bounds and nonlinear constraints

`Bounds` materializes scalar or PyTree lower and upper values against the parameter tree,
broadcasts each bound to its corresponding leaf, and rejects incompatible or inverted
bounds. `ProjectedGradient`, `ProjectedLBFGS`, `ActiveSetNewton`, and
`BoundedNewtonTrustRegion` all preserve the closed box at every accepted iterate.
Projected L-BFGS stores curvature only from accepted projected steps. Active-set Newton
solves over free variables and reports active-set size, primal feasibility,
complementarity, and direction fallbacks. `BoundedNewtonTrustRegion` uses the shared
Steihaug--Toint kernel over a masked Hessian action, detects negative curvature, and
recomputes the local model for the displacement after projection.

`NonlinearConstraint` represents any differentiable residual bounded componentwise by
`lower <= c(x) <= upper`; equal finite bounds are equalities. `AugmentedLagrangian`
normalizes equality and inequality multipliers, increases its penalty only when
feasibility progress is insufficient, and delegates each frozen subproblem to an
unconstrained inner method. `SQP` forms explicit local QP subproblems through
`solve_quadratic_program`. Its optional filter keeps objective and violation dominance
separate; its second-order correction relinearizes constraints at the trial point to
address the Maratos effect. Elastic restoration has a distinct failure status.
`PrimalDualInteriorPoint` is the single native primal-dual method. Its explicit
`mode` selects dense filter, matrix-free centered, matrix-free predictor-corrector,
or exact sparse augmented KKT execution. The dense and matrix-free modes preserve
their original accepted-point, restoration, and transformed-execution contracts.
The sparse mode consumes a prepared `StructuredNonlinearProgram`, retains one
bound-form slack per non-equality constraint component, and reuses one KKT
factorization across affine predictor and centering-corrector right-hand sides.

`prepare_constrained_model` remains the canonical dense equality/lower/upper
model. `SQP(hessian_update=...)` supports damped BFGS, SR1, and exact Lagrangian
Hessians. `FilterGlobalization` owns fixed-capacity objective-feasibility filter
semantics shared by constrained methods.

`plan_kkt` currently declares the actually executed `dense-augmented` form.
`factor_kkt` routes its factorization through `phydrax.linalg`;
`solve_factored_kkt` reuses it for additional right-hand sides. Sparse structured
execution separately uses `plan_sparse_augmented_kkt` and
`assemble_sparse_augmented_kkt`, which compile exact Hessian, Jacobian, barrier
diagonal, regularization, and slack-coupling routes into one full-symmetric
canonical sparse operator.

KKT inertia records its source and whether zero inertia is reliable. Dense
certification uses the Hermitian spectral precision path. A provider-reported
inertia is never silently treated as equivalent evidence. Final optimizer
success remains subordinate to an independently reconstructed KKT certificate.
An internal success whose stationarity, feasibility, regularity, or
complementarity misses tolerance is returned as explicit certification failure.
The current Spineax/cuDSS capability declares zero inertia unreliable, so the
nonconvex structured IPM rejects it as a KKT provider. The sparse-augmented
representation remains executable through a certified dense provider; cuDSS
becomes eligible only when its zero-inertia evidence passes the provider gate.

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

### Structured sparse nonlinear programs

`StructuredNonlinearProgram` is the fixed-topology bound-form optimization IR.
It owns the scalar objective, constraint vector, variable and constraint bounds,
source identities, exact sparse Jacobian plan, optional exact sparse scalar
Lagrangian-Hessian plan, and structure identity.

`prepare_structured_template` freezes dimensions, argument signatures, bound
roles, and derivative topology. `bind_structured_numeric` binds dynamic args,
bound values, and scaling. `refresh_structured_nonlinear` changes numeric values
while rejecting every shape, dtype, finite-role, equality-role, or sparsity
change. `PreparedStructuredNonlinearProgram` is therefore reusable across
fixed-topology parameter families without rediscovering derivatives.

`AbstractStructuredNonlinearMethod` is the method-neutral execution boundary.
`solve_structured_nonlinear` accepts native `PrimalDualInteriorPoint` and
external `IpoptMinimize` implementations without domain-side backend tests.
`StructuredNonlinearResult` contains the generic `MinimizationResult`, exact
structured work, and a portable `StructuredNonlinearWarmStart` containing the
primal, complete constraint multipliers, and direct variable-bound multipliers.

`solve_pooled_structured_nonlinear` advances more independent structured tasks
than execution lanes while retaining input order, exactly-once completion,
failure results, completion placement, utilization, and one execution-signature
identity. Pooling is explicit and never splits mathematically coupled case axes.

`compile_structured_minimization` lowers a fixed-topology PyTree
`MinimizationProblem` into the same IR. `compile_structured_state_design` and
the control compilers reuse this path rather than defining alternate sparse NLP
representations.

`IpoptMinimize` implements the same structured method boundary through low-level
`cyipopt.Problem` callbacks. Install `phydrax[ipopt]` together with an external
Ipopt library, or use conda-forge's `ipopt` and `cyipopt` packages. The adapter
canonicalizes duplicate-free Jacobian coordinates and one lower-triangular
Hessian representative per symmetric pair. Without a Hessian plan it declares
limited-memory mode; an exact plan and an approximation override are mutually
exclusive. No backend is selected by problem size and no failure causes a
fallback.

`StructuredIpoptEvidence` retains mapped/raw status, complete callback and
host/device conversion counts, sparse plan identities, option identity, Hessian
mode, and a typed final `StructuredNonlinearWarmStart`. Warm starts require the
exact program and structure IDs and finite sign-valid bound multipliers. Final
multipliers are normalized into a `ConstrainedOptimalityCertificate`; Ipopt
backend success that misses the requested physical KKT tolerance is returned as
certification failure. This external route declares
`implicit_differentiation=False`; ordinary
`IpoptMinimize.solve(MinimizationProblem, ...)` remains separate.

`structured_solution_jvp` and `structured_solution_vjp` differentiate certified
fixed-active or barrier KKT equations. `structured_parameter_continuation`
exposes a certified fixed-active KKT branch to `phydrax.continuation`.

::: phydrax.optim.StructuredNonlinearProgram

::: phydrax.optim.PreparedStructuredNonlinearProgram

::: phydrax.optim.StructuredNonlinearWarmStart

::: phydrax.optim.StructuredNonlinearResult
::: phydrax.optim.IpoptCallbackCounts

::: phydrax.optim.IpoptStatusEvidence

::: phydrax.optim.StructuredIpoptEvidence

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
`PrimalDualInteriorPoint` return the multiplier, slack, activity, and
stationarity certificate required by this path.

The constrained derivative is classical only at a successful regular KKT point with a
strictly complementary active set. A failed primal solve, ambiguous activity, or
singular active KKT matrix raises an explicit runtime error; Phydrax does not choose a
subgradient at an active-set transition. The solution map supports forward and reverse
mode, filtered JIT, nested parameter and argument PyTrees, and batching. Constraint
topology and bound roles remain static across transformed calls.

`constrained_solution_jvp` and `constrained_solution_vjp` expose both
`fixed-active` and `barrier` derivative systems with condition and active-set
evidence. The fixed-active route uses only the certified active inequalities;
the barrier route differentiates positive slack-multiplier complementarity.

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

::: phydrax.optim.DenseNewtonDogleg

---

::: phydrax.optim.SteihaugToint

---

::: phydrax.optim.TrustRegionQuadraticProblem

---

::: phydrax.optim.solve_trust_region_subproblem

---

::: phydrax.optim.BoundedNewtonTrustRegion

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

::: phydrax.optim.BoundedGaussNewton

---

::: phydrax.optim.BoundedLevenbergMarquardt

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

::: phydrax.optim.PrimalDualInteriorPoint

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


## Canonical linear and quadratic programs

`LinearProgram` and `QuadraticProgram` share one audited mathematical-programming
contract. Both accept optional equalities `Ax = b`, inequalities `Gx ≤ h`, and
native `Bounds`. Bounds do not appear on the public equality or inequality axes:
their lower and upper multipliers are reported separately. Fixed bounds are treated
as equalities internally without exposing synthetic rows.

`LinearProgram` stores no dense zero Hessian. A selected dense QP method creates its
temporary zero quadratic only at execution; sparse LP backends can consume the linear
form directly.

!!! example
    ```python
    import jax.numpy as jnp
    import phydrax as phx


    lp = phx.optim.LinearProgram(
        jnp.asarray([-1.0, -0.5]),
        inequality_matrix=jnp.asarray([[1.0, 1.0]]),
        inequality_rhs=jnp.asarray([1.0]),
        bounds=phx.optim.Bounds(0.0, jnp.inf),
        problem_id="bounded-lp",
    )
    lp_result = phx.optim.solve_linear_program(lp)
    assert lp_result.status == phx.optim.ConvexProgramStatus.OPTIMAL

    qp = phx.optim.QuadraticProgram(
        jnp.asarray([[2.0, 0.0], [0.0, 2.0]]),
        jnp.asarray([-2.0, -5.0]),
        equality_matrix=jnp.asarray([[1.0, 1.0]]),
        equality_rhs=jnp.asarray([2.0]),
        bounds=phx.optim.Bounds(0.0, jnp.inf),
        problem_id="bounded-qp",
    )
    policy = phx.optim.ConvexSolvePolicy(
        phx.optim.DensePrimalDualQP(),
        termination=phx.optim.ConvexTermination(absolute=1e-7),
    )
    qp_result = phx.optim.solve_quadratic_program(qp, policy=policy)
    assert qp_result.successful
    ```

Convexity remains an explicit mathematical contract. Array-valued `Q` is symmetrized
and carries asserted positive-semidefinite evidence; Phydrax never projects an
indefinite matrix onto the PSD cone. Control compilers attach verified evidence after
their explicit cost checks.

### Policies and methods

`ConvexSolvePolicy` separates the method, termination thresholds, regularization,
materialization permission, resource budgets, and failure mode. Available methods are:

- `DensePrimalDualQP`, the native dense predictor-corrector method;
- `QPaxInteriorPoint`, QPax 0.1.4's dense public implicit backend;
- optional `MPAXraPDHG`, restarted-average PDHG for assembled LP/QP;
- optional `MPAXr2HPDHG`, reflected restarted Halpern PDHG for LP.

No method fallback occurs. Dense resource rejection is a terminal configuration error,
not a request to select another backend.

Dense planning and direct QP solves check the assembled KKT entry count,
factorization bytes, and workspace estimate against the declared limits before
execution. `FailurePolicy("status")` returns audited terminal evidence;
`FailurePolicy("error")` raises when that audit is nonoptimal. These policies never
select a different method.

`ConvexTermination.absolute + relative * scale` is the original-data KKT audit
threshold; providers that expose relative stopping receive both components.
`primal_infeasible` and
`dual_infeasible` are direct tolerances for independently audited dual and primal
rays; they are not inferred from a provider status.

Native and QPax methods solve with the explicit regularized Hessian
`Q + regularization * I`; the result audits that equation separately from stationarity
for the original `Q`. QPax fixes its fraction-to-boundary multiplier and therefore has
no configurable `step_fraction` field. MPAX retains its scaling and first-order
iteration evidence, while Phydrax re-audits the original unscaled program.

### Prepared lifecycle and warm starts

Repeated programs use:

1. `plan_convex_program`;
2. `prepare_convex_template`;
3. `bind_convex_numeric`;
4. `refresh_convex_program`;
5. `solve_convex_program`.

Refresh preserves dimensions, bound roles, constraint topology, dtype, batch shape,
method, and problem identity. It changes only numeric coefficients, increments
`numeric_version`, and creates a new runtime-unique `numeric_binding_id`.

The returned execution and provenance record the public program's `problem_id` and
`structure_id`, exact policy fingerprint, prepared numeric version, and numeric binding
identity. The binding identity distinguishes independently prepared programs that have
the same structure, policy, and version. Lowering an LP to a dense QP or conic provider
representation does not leak that internal identity.

`ConvexWarmStart` stores primal, equality-dual, inequality-dual, slack, and bound-dual
arrays for one exact `structure_id`. The native dense method requires strictly positive
inequality slacks and multipliers. `ConvexWarmStart.from_result` explicitly
interiorizes dual/slack arrays; a primal on an active variable bound still requires a
problem-aware shift into the interior. QPax rejects warm starts before backend
execution.

### Differentiation

`solve_quadratic_program_primal` is the differentiable, primal-only surface.
`ConvexDifferentiationPolicy("active-set-kkt")` differentiates the locally fixed active
KKT system. `ConvexDifferentiationPolicy("backend-implicit")` requires
`QPaxInteriorPoint` and calls QPax's public implicit custom VJP. MPAX exposes only
explicitly requested algorithmic differentiation: the selected method must use
`unroll=True` and the differentiation policy must be `"algorithmic"`.

The active-set derivative is valid only at a successful regular KKT point with an
unambiguous strictly complementary active set. Invalid forward solves or singular and
ambiguous active systems produce explicit failure/NaN sensitivities rather than a
fabricated subgradient.

### Results, audits, and certificates

`ConvexProgramResult` retains:

- primal variables;
- user equality and inequality multipliers/slacks;
- lower and upper bound multipliers;
- original and regularized stationarity;
- feasibility, complementarity, and KKT residuals;
- backend convergence and iteration evidence;
- `ConvexProgramCertificate`;
- `ConvexProgramProvenance`.

Public constraint arrays never include synthetic bound rows. The independent audit does
include native bounds when computing primal feasibility and stationarity.

`ConvexProgramStatus` distinguishes `OPTIMAL`, `ITERATION_LIMIT`,
`PRIMAL_INFEASIBLE`, `DUAL_INFEASIBLE`, `NONFINITE_INPUT`, `NONFINITE_OUTPUT`,
`NUMERICAL_FAILURE`, `BACKEND_FAILED`, and `INVALID_PROBLEM`. Infeasible or unbounded
statuses require an independently validated dual or primal ray. A backend status alone
is never promoted.

::: phydrax.optim.LinearProgram
    options:
        members:
            - __init__

---

::: phydrax.optim.QuadraticProgram
    options:
        members:
            - __init__

---

::: phydrax.optim.ConvexProgramResult
    options:
        members:
            - successful

---

::: phydrax.optim.ConvexSolvePolicy

---

::: phydrax.optim.ConvexTermination

---

::: phydrax.optim.DensePrimalDualQP

---

::: phydrax.optim.QPaxInteriorPoint

---

::: phydrax.optim.prepare_convex_program

---

::: phydrax.optim.refresh_convex_program

---

::: phydrax.optim.solve_convex_program

---

::: phydrax.optim.solve_linear_program

---

::: phydrax.optim.solve_quadratic_program

---

::: phydrax.optim.solve_quadratic_program_primal

## Canonical conic programs

`ConicProgram` represents
`min 0.5 xᵀPx + qᵀx` subject to `Ax + s = b`, `s ∈ K`, and native variable
bounds. The public cone product supports:

- `ZeroCone`;
- `NonnegativeCone`;
- `SecondOrderCone`;
- `RotatedSecondOrderCone`;
- `PositiveSemidefiniteCone`;
- `ExponentialCone`;
- `PowerCone`;
- `ProductCone`.

Every cone exposes primal/dual projection, membership residual, interior margin,
dual-projection smoothness margin, and complementarity. Cone topology is static across
program batches and numeric refreshes.

`PositiveSemidefiniteCone(matrix_size)` uses Clarabel's Frobenius-isometric `svec`
coordinates: the upper triangle is stacked by column, diagonal entries are unchanged,
and off-diagonal entries are multiplied by `sqrt(2)`. For a three-by-three matrix the
order is `(00, 01, 11, 02, 12, 22)`. `pack` rejects materially nonsymmetric real
matrices; `unpack` reconstructs the symmetric matrix. Projection clips matrix
eigenvalues at zero and uses a stable spectral Fréchet rule. Repeated nonzero
eigenvalues are regular; any zero eigenvalue makes the ordinary projection derivative
unavailable.

`ExponentialCone()` has canonical coordinates `(x, y, z)` and the closed primal region
`y > 0, y exp(x/y) <= z`, together with the face `y = 0, x <= 0, z >= 0`.
`PowerCone(exponent)` has coordinates `(x, y, z)`, static
`0 < exponent < 1`, and constraint
`x^exponent y^(1-exponent) >= abs(z)` with nonnegative `x` and `y`.
Both asymmetric cones derive dual projection from Moreau decomposition,
`project_dual(v) = v + project(-v)`.
EXP membership and boundary products are evaluated in log form,
`log(y) + x/y`, before reconstructing a representable product. This preserves valid
float64 points with tiny positive `y` and ratios beyond the direct `exp` overflow
threshold; unrepresentable products remain explicit infinity/failure rather than a
clipped finite value.
For float32 POW projection, primal/polar classification is evaluated in original
coordinates at float64 working precision. General inputs are promoted before
normalization, so normalization preserves every representable float32 coordinate; the
scalar root and KKT derivative are then formed in normalized float64 coordinates and
cast back to float32. The normalized scalar and cone-boundary residual is bounded by
`512 * eps`. Qualification compares against a 100-digit reference over exponents
`0.01` through `0.99`, positive scales `1e-30` through `1e30`, and mixed-magnitude
coordinates spanning `1e-30` through `1e30`, with relative projection, scaled
feasibility, and JVP error bounded by `5e-5`. This is a bounded numerical contract, not
a uniform high-accuracy claim at every exponent/input combination.

The EXP and POW projectors use JAX-native, fixed-topology safeguarded
Newton-bisection roots. An accepted root must retain a sign bracket and satisfy
scale-aware residual and bracket-width evidence. Failed roots return nonfinite
projections, causing membership audits and conic sensitivities to fail closed. Their
custom JVPs differentiate the locally regular projection KKT system rather than the
finite root iteration schedule. Axis, face, spectral-zero, and region-transition points
have zero smoothness margin and are not ordinary differentiability claims.

Clarabel 0.11.1 is an optional explicit host backend selected with
`ConvexSolvePolicy(ClarabelInteriorPoint())`. PSD, EXP, and POW blocks map directly to
the provider's native coordinates; rotated cones alone use a documented SOC isometry.
Native variable bounds are represented separately in the public result. Clarabel is
not a JIT or differentiation boundary and is never selected automatically. Its public
Phydrax program surface is dense even though the adapter passes assembled CSC matrices
to the host provider.

### Regular conic primal sensitivity

`prepare_conic_sensitivity` binds one `PreparedConvexProgram` to the exact
`ConvexProgramExecution` produced from its current `numeric_version`. It rejects stale
or structurally different executions before forming derivatives. The prepared object
contains only JAX numerical state and static cone topology; the Clarabel provider does
not enter the derivative kernel.

The derivative is defined by the direct projection-KKT residual. With
`s = b - Ax` and dual variable `y`, conic complementarity is equivalent to
`y = project_dual(y - s)`. Phydrax therefore differentiates

`H(x, y) = (P_effective x + q + Aᵀy, y - project_dual(y + Ax - b))`,

where `P_effective = P + λI` includes the solve policy's explicit regularization.
One solve with `D H` computes a JVP; one adjoint solve with `(D H)ᵀ` computes a VJP.
No homogeneous scale variable or solver iteration is differentiated.

`ConicProgramData` is the fixed-topology numerical data space. It contains tangents or
cotangents for `P`, `q`, `A`, `b`, lower bounds, and upper bounds. Cone identity and
block topology remain static. `conic_primal_jvp` maps one `ConicProgramData` tangent
to a primal tangent. `conic_primal_vjp` maps a primal cotangent back to
`ConicProgramData`.

The ordinary derivative is available only when:

- the audited forward result is optimal and finite;
- the projection point is separated from every cone projection kink;
- the projection-KKT Jacobian is numerically full rank;
- the selected linear or adjoint solve converges with finite condition evidence.

Weak complementarity, an SOC apex or transition surface, a zero PSD eigenvalue, an
EXP/POW axis or projection-region transition, nonunique primal-dual roots,
infeasibility, and failed projection or linear solves return `regular=False` and NaN
sensitivity values in status mode. Error-mode linear policies raise instead. Phydrax
does not silently return a selected generalized derivative.

Native bounds are lowered into fixed, finite-lower, and finite-upper cone rows. A fixed
bound has one valid tangent, so JVP inputs require equal lower and upper perturbations.
The VJP splits its cotangent equally between the two public bound arrays. Infinite
bounds accept only zero tangents and receive zero cotangents.

The derivative solve currently requires undamped `DenseSVD`, so full-rank and condition
evidence are available before any derivative is accepted. Tikhonov damping would change
the derivative equation and is rejected; regularize the executed convex program through
`ConvexSolvePolicy.regularization` instead. Existing materialization and resource
budgets bound this dense path. Matrix-free sensitivity is deferred until its regularity
contract can certify more than convergence and a small residual.

Preparation and Clarabel execution remain host operations. Once prepared,
`conic_primal_jvp` and `conic_primal_vjp` are JIT-compatible JAX kernels, but this does
not make the forward Clarabel solve device-resident. The API provides first-order
mathematical solution sensitivity only; higher-order differentiation is unsupported.

```text
import equinox
import phydrax

prepared = phydrax.optim.prepare_convex_program(problem, policy)
execution = phydrax.optim.solve_convex_program(prepared)
sensitivity = phydrax.optim.prepare_conic_sensitivity(prepared, execution)

tangent = phydrax.optim.ConicProgramData.zeros_like(problem)
tangent = equinox.tree_at(
    lambda data: data.constraint_rhs,
    tangent,
    rhs_direction,
)
forward = phydrax.optim.conic_primal_jvp(sensitivity, tangent)
reverse = phydrax.optim.conic_primal_vjp(sensitivity, primal_cotangent)
```

::: phydrax.optim.ConicProgram

---

::: phydrax.optim.ConicProgramData

---

::: phydrax.optim.PreparedConicSensitivity

---

::: phydrax.optim.ConicSensitivityResult

---

::: phydrax.optim.ZeroCone

---

::: phydrax.optim.NonnegativeCone

---

::: phydrax.optim.SecondOrderCone

---

::: phydrax.optim.RotatedSecondOrderCone

---

::: phydrax.optim.PositiveSemidefiniteCone

---

::: phydrax.optim.ExponentialCone

---

::: phydrax.optim.PowerCone

---

::: phydrax.optim.ProductCone

---

::: phydrax.optim.ClarabelInteriorPoint

---

::: phydrax.optim.solve_conic_program

---

::: phydrax.optim.prepare_conic_sensitivity

---

::: phydrax.optim.conic_primal_jvp

---

::: phydrax.optim.conic_primal_vjp


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

### Model-based and multistart search

`BOBYQA` and `COBYQA` share the same quadratic interpolation, poisedness, trust
region, and polling kernel. BOBYQA accepts bounds only; COBYQA evaluates the
physical objective plus explicit feasibility merit and independently reports
the final physical objective and feasibility. Neither method claims global
optimality.

`MultiStartPolicy` deterministically generates bounded-uniform or normal starts,
divides the declared work budget across local solves, retains every status and
objective, and returns the best certified local result. `POUNDERS` is the
residual-wise counterpart for black-box least squares.

`nonlinear_peer_manifest.json` freezes source revisions and exact runtime
identities. Peer runners reject runtime-identity, revision, and
initial-fingerprint mismatches before comparison.
`best_nonlinear_campaigns.py` writes ordinary flat JSON rows. Backend outcomes
and independent mathematical certificates remain separate; unavailable
evidence uses `null` rather than non-finite sentinels, and cold, warmup, and
repeated steady timing remain distinct. Performance profiles are formed per
family and compatible work unit. Global rows certify a known global target gap
rather than relabeling local stationarity as global evidence.

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
| `SciPyMinimize` | Yes, host-only | No | No | No |
| `NLoptMinimize` | Yes when NLopt is installed | No | No | No |
| `IpoptMinimize` | Yes when cyipopt is installed | No | No | No |
| `ceres_least_squares` | Explicit callback boundary | No | No | No |

`FunctionalSolver.solve` accepts standard and extra-argument Optax transformations,
Phydrax KFAC and Riemannian optimizers, and native scalar, least-squares, and composite
iterative methods. Operator fitting accepts supplied Optax transformations.
Resumable `fit_operator` runs require a stable `optimizer_id` whenever the transformation
is supplied externally so checkpoint identity does not depend on an opaque Python object.

An explicit `ParameterSubspace` may be supplied to `FunctionalSolver.solve` or
`fit_operator`. In the initial contract this restriction is supported only by
standard and extra-argument Optax transformations. KFAC, Evosax, mirror,
Riemannian, scalar, least-squares, and composite backends reject it rather than
silently optimizing the complete ambient PyTree.

Evosax distribution-based algorithms remain accepted by `FunctionalSolver`; its
population-based algorithms require an explicit finite search-space contract and are
rejected there. Optimistix interoperation is deliberately explicit and standalone:
wrap a compatible upstream minimizer in `OptimistixMethod`. Phydrax does not inspect an
arbitrary upstream object or silently reinterpret its stopping rules.
