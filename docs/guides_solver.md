# Solvers and training

This guide explains how `FunctionalSolver` evaluates losses and how `solve()` updates parameters.

## Functional minimization versus trajectory integration

`FunctionalSolver` and the differential backend solve different problems:

| Path | Computes | Typical use |
| --- | --- | --- |
| `FunctionalSolver` | Parameters minimizing residual, data, integral, and model-loss terms | PINNs, inverse problems, variational objectives |
| `solve_diffrax` | One finite-dimensional ODE/SDE trajectory | Numerical reference solves, differentiable simulation |
| `solve_diffrax_ensemble` | Coupled SDE trajectories with retained realization provenance | Process uncertainty, stochastic transition data |

An SDE is specified by `DifferentialProblem` with one or more named `WienerTerm`
objects and an explicit `phx.stochastic.WienerRealization`. The problem declares
coefficient shapes and Itô/Stratonovich semantics. The realization owns a global
support interval, Brownian root key, sample shape, approximation tolerance,
Lévy-area level, and optional noise-basis identity. Reusing it over subintervals,
models, or discretizations queries the same global paths.

The default Itô and Stratonovich methods are fixed-step Euler--Maruyama and
Euler--Heun, so `dt0` is required. Phydrax rejects interpretation, basis, support,
fixed-step tolerance, and Lévy-area mismatches before entering Diffrax. Pass native
Diffrax solver, controller, event, and adjoint objects when a different method is
needed.

Pass `dense=True` to either integration function when states must be evaluated between
saved times. `DifferentialSolution.evaluate(query_times)` accepts arbitrarily shaped
shared query arrays and returns `sample_shape + query_shape + state_shape`; dense data
is not retained by default.

### Array state geometry

`DifferentialProblem(state_geometry=...)` keeps the usual vector-field signature
`(time, state, args) -> state-shaped array`, while declaring that the array lies
on a constrained state space. Metrix supplies Euclidean, embedded, pointwise,
special-orthogonal, and symmetric-positive-definite geometries. A nontrivial
geometry validates the initial state and requires a geometric solver:

```python
import jax.numpy as jnp
import phydrax as phx

geometry = phx.metrix.SpecialOrthogonalStateGeometry(3)
omega = jnp.array(
    [[0.0, -0.4, 0.2], [0.4, 0.0, -0.1], [-0.2, 0.1, 0.0]]
)
problem = phx.solver.DifferentialProblem(
    lambda t, rotation, args: rotation @ omega,
    jnp.eye(3),
    t0=0.0,
    t1=2.0,
    state_geometry=geometry,
)
solution = phx.solver.solve_diffrax(
    problem,
    save_times=jnp.linspace(0.0, 2.0, 21),
    solver=phx.solver.RKMK(geometry),
    dt0=0.05,
    dense=True,
)
```

`GeometricEuler` and `RKMK` integrate deterministic dynamics. RKMK requires an
exact local pullback; an `EmbeddedStateGeometry` must therefore supply explicit
inverse-retraction and pullback callables. `CommutatorFreeSolver` executes an
explicit `CommutatorFreeTableau` only when the geometry declares a shared
trivialization (the built-in Euclidean and SO(n) geometries). SPD(n) uses RKMK
rather than the commutator-free solver. `SRKMK` is the explicit Stratonovich
solver for nontrivial geometry. Generic Itô geometry is intentionally rejected
rather than approximated by projection.

SO(n) supports exponential and Cayley retractions. The exponential
`inverse_retract` uses a degree-63 Cayley/atanh series and accepts only a Cayley
spectral radius strictly below 0.5; it explicitly rejects points outside that
numerically justified local neighborhood. The Cayley pullback uses its analytic
inverse differential; the exponential pullback solves every leading batch
element independently, normalizes its right-hand side, and applies a
differentiable matrix-free solve of the exponential JVP with zero absolute
tolerance, dimension-scaled Krylov cycles, and fixed \(O(n^2)\) workspace per
state. It recomputes each relative differential residual and rejects a
nonconverged solve. Neither solver pullback depends on the logarithm cutoff.
SPD(n) uses a symmetric
congruence/exponential retraction. Dense
interpolation and root-finding events evaluate through the same retraction, so
queried states remain on the declared state space.
`DifferentialSolution.state_geometry_id`, `solver_id`, and `resolved_method`
record geometry and numerical-method provenance.


`DifferentialSolution.to_predictive()` converts an ensemble to a
`PredictiveField` whose leading sample axis is labeled `process`. This label means
intrinsic stochastic forcing. Time-step or spatial-discretization error is
`numerical` uncertainty and is not estimated or inserted automatically. Spatial
SDE/SPDE models must expose a finite-dimensional semidiscretization and a declared
noise basis.

See [API → Solver → Differential equation integration](api/solver/differential.md)
for the complete shape, replay, and result contract.

## What `FunctionalSolver` does

A `FunctionalSolver` is a lightweight orchestrator that holds:

- `functions`: a mapping `{name: DomainFunction}` of the current fields,
- `terms`: one ordered collection of training penalties and signed objectives,
- `evaluation_terms`: optional held-out terms used only for diagnostics and logging,
- model-level losses attached to models with `model.add_model_loss(...)` or a custom
  model `__loss__` hook, and
- optional `enforcement`: a compiled `EnforcementProgram` that replaces raw fields
  with ansatz functions satisfying selected conditions.

The training functional is the sum of training-term losses and attached model
losses:

$$
\mathcal J = \sum_i \ell_i + \sum_k r_k.
$$

`evaluation_terms` are evaluated against the same current ansatz functions, but
they do not contribute to `loss(...)`, gradients, optimizer state, or best-model
selection. Use them for validation folds, held-out cases, or non-training
diagnostics.

## Loss evaluation (`loss(...)`)

When you call `solver.loss(key=...)`:

1) If an enforcement program is configured, it transforms the current `functions`
   mapping into *ansatz functions* via `solver.ansatz_functions()`.
2) The provided PRNG key is split into independent subkeys for terms and model losses.
3) Each training term is evaluated and the scalar losses are summed.
4) Model-level losses attached to the raw trainable models are evaluated and added.

Additional keyword arguments are forwarded to each term's `.loss(...)`.
The `iter_` keyword, when present, is also forwarded to model losses.

Raw objective terms may be negative. `IntegralFunctional` integrates its density
without squaring it; use this for energy/Ritz minimization, not for a residual penalty.

### Sampled term materialization

Terms derived from `AbstractSamplingTerm` own two stages:

1. `sample(key=...)` materializes an immutable batch of paths, labels, particles, or
   derivative probes;
2. `loss(..., batch=batch)` evaluates the current model on that batch.

For Optax training, `FunctionalSolver` performs stage 1 once per optimizer update,
outside `filter_value_and_grad` and the compiled loss. The identical batch is reused
for every value, gradient, line-search, and same-update parameter-view evaluation.
This prevents differentiation through Monte Carlo target construction and prevents
an optimizer population from silently receiving different targets within one update.
Evosax likewise materializes one batch per population update.

Fixed sampled terms return the same batch on every call and are the default for
common-random-number comparisons. Resampled terms receive deterministic fresh
subkeys each update. Multiple sampled terms receive distinct subkeys. Keep probe
counts, path counts, and other batch-shape policy fields fixed during one JIT-compiled
run.

Some sampled losses are signed estimators of nonnegative mathematical quantities. In
particular, a randomized-residual U-statistic can be negative on one finite batch even
though its expectation is a squared residual. Do not use the noisiest training draw
for `keep_best` selection. Prefer fixed probes for deterministic optimization or an
independent validation objective whose estimator matches the model-selection
criterion.

## Model losses

Use model losses for parameter-space penalties that are not residuals over a domain,
such as spectral penalties, norm targets, sparsity penalties, or architecture-specific
regularization. These losses are evaluated once per distinct raw model in
`solver.functions`; if two fields share the same model object, its model loss is not
double-counted.

For existing Phydrax models, attach a scalar penalty with `add_model_loss(...)`:

```python
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import phydrax as phx

model = phx.nn.models.MLP(
    in_size=2,
    out_size="scalar",
    width_size=32,
    depth=2,
    key=jr.key(0),
).add_model_loss(
    lambda m: (jnp.linalg.norm(m.layers[0].weight) - 1.0) ** 2,
    weight=1e-4,
    label="unit_weight_norm",
)
```

The penalty callable receives the wrapped model as its first argument and may accept
keyword-only `key` and `iter_` arguments.

Custom model classes can instead implement `__loss__`:

```python
class MyModel(eqx.Module):
    weight: jax.Array

    def __call__(self, x, *, key=None):
        return self.weight @ x

    def __loss__(self, *, key=None, iter_=None):
        return 1e-4 * jnp.sum(self.weight**2)
```

During `solve(...)`, model losses contribute to gradients, optimizer state, and
best-model selection. When `log_terms=True`, text logs print them as
`[model i] ...`, and TensorBoard writes them under `train/model_losses/...`.

## Exact enforcement

Exact enforcement is optional, but common when boundary, initial, or interior
conditions must hold by construction rather than through a penalty. Declare each
hard condition as an `EnforcementSpec`, compile the specifications once, and pass
the resulting `EnforcementProgram` to the solver:

```python
geom = phx.domain.Interval1d(-1.0, 1.0)
u = geom.Function("x")(lambda x: x[0])
functions = {"u": u}

component = geom.component()
interior_condition = phx.conditions.Residual(
    "u", component, lambda value: value
)
terms = (
    phx.terms.ResidualPenalty(
        interior_condition,
        phx.integration.per_step(
            phx.integration.mean_over(component),
            phx.domain.PointSampling(16),
        ),
    ),
)
boundary = geom.component({"x": phx.domain.Boundary()})
boundary_condition = phx.conditions.Dirichlet("u", boundary, target=0.0)
specs = (phx.enforcement.EnforcementSpec(boundary_condition),)
options = phx.enforcement.EnforcementOptions(gate_method="auto")
program = phx.enforcement.compile(
    functions,
    specs,
    options=options,
    key=jr.key(0),
)
solver = phx.solver.FunctionalSolver(
    functions=functions,
    terms=terms,
    enforcement=program,
)
```

The program is applied before scalar terms are evaluated, so every residual and
objective sees the same enforced fields. When there are no hard specifications
or interior anchors, pass `enforcement=None` rather than compiling an empty
program.

Geometry Dirichlet ansätze and preservation overlays use a dimensionless gate
derived from the certified boundary field. Set `gate_method` on
`EnforcementOptions`: `"auto"` selects an exact domain-specific profile for
intervals and hyperrectangles and the compact field transform for general
compiled geometry, `"global_r_equivalence"` selects the broad generic
transform, and `"compact"` makes the fallback explicit.
`gate_saturation_fraction` and `gate_linear_fraction` configure only the compact
transform.

Neumann, Robin, Sommerfeld, and traction ansätze use a compact dimensional
factor whose outward boundary derivative is one. Certified level sets are first
locally normalized by their gradient magnitude; signed-distance fields already
have the required local jet. A high-order compact saturation is exactly linear
near the boundary and bounds the interior amplitude. The factor gradient is the
outward unit normal at regular boundary points and inherits the field
certificate's piecewise regularity elsewhere.

`EnforcementSpec` infers derivative requirements from standard Neumann, Robin,
absorbing, and initial conditions. A custom transform must declare its
`DerivativeRequirement` values explicitly. The compiler uses these orders when
constructing later initial and interior overlays, preserving boundary
derivatives through the declared order.

See [API → Solver → Exact enforcement](api/solver/enforcement.md) for
the compiler, specification, anchor, and program APIs.

## Training (`solve(...)`)

`FunctionalSolver.solve(...)` runs an optimization loop over the parameters contained inside
`solver.functions`. Under the hood it uses a Phydrax-aware Equinox partition to split
the function PyTree into:

- **trainable parameters**: inexact arrays inside trainable models/functions,
- **non-trainable state**: domains, observed data tables, fixed trajectory signals,
  hard-enforcement lookup tables, integer metadata, and other fixed state.

This distinction matters for physics-data problems. Observed data may be a JAX
array so it can participate in JIT-compiled residuals, but it is not an optimizer
parameter and is excluded from gradients, optimizer state, and weight decay.
Explicit learnable fields created through model wrappers or `Domain.Parameter(...)`
remain trainable. Literal constant `DomainFunction` values are treated as fixed
state; use `Domain.Parameter(...)` when a scalar/vector coefficient should be
optimized.

Use `solver.partition_functions()` to inspect the exact split, or
`solver.trainable_functions()` when an external optimizer needs the trainable PyTree
shape.

### Optimizer support

`optim=` can be:

- an Optax `GradientTransformation` (standard first-order optimizers),
- an Optax `GradientTransformationExtraArgs` (line-search style optimizers), or
- an Evosax distribution-based algorithm.

Evosax population-based algorithms are not accepted by `FunctionalSolver`. They
require an explicit initial population, finite search bounds, and selection semantics
that a general neural-network parameter PyTree does not provide. Low-dimensional
geometry design uses
[`DesignConstraintSystem.search`](api/geometry.md#bounded-global-design-search),
which supplies those contracts explicitly.

### Optimizer evaluation parameters

Some Optax transformations update raw training parameters but prescribe a different
parameter view for validation and model selection. Pass that transformation as
`evaluation_parameters(state, training_parameters)`. Gradients and optimizer updates
continue to use the raw parameters; diagnostics, best-state selection, and returned
functions use the transformed view. The transform must preserve the parameter PyTree,
leaf shapes, and dtypes.

```py
import optax

optimizer = optax.contrib.schedule_free(optax.sgd(1e-3), 1e-3)
solver = solver.solve(
    num_iter=1000,
    optim=optimizer,
    evaluation_parameters=optax.contrib.schedule_free_eval_params,
)
```

This contract is available for Optax optimizers, not evosax algorithms.

### Iteration counter (`iter_`)

During training, the current epoch index is passed to each term's loss as `iter_`
(as a JAX scalar), so terms can implement schedules such as annealed weights.

### Adaptive collocation sources

Sampling policy belongs to the integration source, not to a condition.
`phx.integration.adaptive(target, initial_plan, policy)` constructs an
`AdaptiveIntegration` source. Use it with `ResidualPenalty`; the residual condition
continues to describe only the fields, component, and operator:

```python
component = geom.component()
condition = phx.conditions.Residual(
    "u",
    component,
    lambda f: f,
)
```

Use `phx.sampling.collocation.RECOMMENDED_COLLOCATION_DEFAULTS` to inspect that
contract. When adaptive refinement is explicitly requested, R3 is the supported
general starting point. Support is declared by `COLLOCATION_POLICY_SUPPORT` and
`collocation_policy_support(...)`:

- **Stable:** fixed scrambled Sobol, periodic replacement, R3, and periodic
  separable sampling.
- **Conditional:** coreset collocation for residual-weighted, diversity-preserving
  paired points; RAR-D for sufficiently resolved oscillatory residuals; and
  hierarchical-axis refinement for nested coordinate-separable discretizations.

For production-style adaptive runs, separate proposal generation from solver
control with `controlled_collocation(...)`:

```python
policy = phx.sampling.collocation.controlled_collocation(
    phx.sampling.collocation.R3(
        refresh_every=25,
        sampler="sobol_scrambled",
        min_replace_fraction=0.1,
        max_retain_fraction=0.9,
    ),
    schedule=phx.sampling.collocation.RefreshSchedule(25),
    monitor=phx.sampling.collocation.ResidualMonitor(
        sampler="sobol_scrambled"
    ),
    guard=phx.sampling.collocation.RefreshGuard(
        max_relative_regression=0.0,
        max_consecutive_rejections=2,
        suspension_steps=100,
    ),
    budget=phx.sampling.collocation.AdaptationBudget(
        max_candidate_evaluations=100_000,
        max_monitor_evaluations=25_000,
    ),
    anchors=phx.sampling.collocation.CoverageAnchors(0.25),
)
source = phx.integration.adaptive(
    phx.integration.mean_over(component),
    phx.domain.PointSampling(2_048, design="sobol_scrambled"),
    policy,
)
term = phx.terms.ResidualPenalty(condition, source)
solver = phx.solver.FunctionalSolver(functions={"u": u}, terms=[term])
```

The initial plan is a `PointSampling` or `GridSampling` plan. R3, RAR-D, and
periodic point replacement require a `PointSampling` plan with one integer sample
count. Adaptive refinement is opt-in. For fixed collocation, use an ordinary
per-step source and state the design directly:

```python
source = phx.integration.per_step(
    phx.integration.mean_over(component),
    phx.domain.PointSampling(2_048, design="sobol_scrambled"),
)
term = phx.terms.ResidualPenalty(condition, source)
```

`FunctionalSolver` detects training `ResidualPenalty` terms backed by
`AdaptiveIntegration`. It initializes one immutable population per such term,
refreshes eligible populations before optimizer updates, and materializes the
current batch and its local active weights as an integration realization for loss
evaluation. The trained solver retains those populations, so another `solve(...)`
call continues from the same state. Adaptive sources are not allowed in
`evaluation_terms`.

Controlled collocation maintains three distinct populations:

1. the mutable training population proposed by the wrapped policy,
2. a fixed independent monitor population used for acceptance and rollback,
3. an untouched test population owned by the application, never by the solver.

A proposed population is evaluated on the next eligible control step. Monitor
regression rejects and rolls it back; repeated rejection suspends adaptation for
the configured cooldown. `FunctionalSolver.solve(...)` settles any proposal still
pending after the final optimizer step without admitting another proposal. Coverage
anchors reserve a persistent low-discrepancy fraction of paired populations so
residual concentration cannot consume all global coverage. Monitor budgets reserve
the next validation evaluation before a proposal is admitted, so exhausting a
candidate or monitor budget cannot leave the final population unvalidated.

Use coreset collocation when an underresolved or localized residual repeatedly
concentrates ordinary residual-only selection. Delay selection until the network
residual is informative, retain an explicit global-coverage stratum, and validate
proposals independently:

```python
policy = phx.sampling.collocation.controlled_collocation(
    phx.sampling.collocation.CoresetCollocation(
        refresh_every=25,
        start_at=100,
        sampler="halton_scrambled",
        candidate_multiplier=8,
        exponent=0.5,
        uniform_fraction=0.5,
        minimum_ess_fraction=0.5,
        max_fill_distance_ratio=3.0,
    ),
    anchors=phx.sampling.collocation.CoverageAnchors(0.25),
)
```

Each refresh scores `candidate_multiplier × retained_count` points. `exponent=0.5`
turns the usual squared pointwise residual into residual-magnitude importance. The
score distribution is normalized before mixing it with the uniform distribution, so
`uniform_fraction` is dimensionless and does not depend on PDE units. If the mixture
would have less than `minimum_ess_fraction × candidate_count` effective samples, the
policy increases the effective uniform fraction and reports the adjustment.

Candidate coordinates are range-normalized on every refresh. With `kernel=None` (the
default), the policy constructs a shared `SquaredExponentialKernel` whose length
scale is the median nonzero pairwise distance of a bounded candidate subsample. Pass
any explicit `phx.kernels.AbstractPositiveDefiniteKernel` to use the same covariance
semantics as GP inference; for example,
`phx.kernels.Matern32Kernel(length_scale=0.25)` uses normalized-coordinate units.
The fill-distance guard rejects a proposal whose normalized candidate fill distance
exceeds `max_fill_distance_ratio` times that of the current population. Selection
validity, acceptance, MMD, requested and effective mixture fractions, importance ESS,
resolved stationary scales, fill distances, guard activation, candidate count, and
kernel-evaluation count are all exposed through solver data metrics.

The policy uses the ordinary `ControlledCollocationPolicy` monitor, rollback, budget,
and `CoverageAnchors` contracts. Its residual-derived sampling measure intentionally
changes the effective training distribution, so keep it conditional and compare it
against fixed low-discrepancy collocation on an untouched monitor and the physical
quantity of interest.

Choose the population representation first:

| Representation | Policies | Budget unit | Appropriate support |
| --- | --- | --- | --- |
| `PointBatch` | `PeriodicCollocation`, `R3`, `RARD`, `CoresetCollocation` | retained points and residual candidate evaluations | General pointwise PINNs |
| `GridBatch` | `PeriodicSeparableCollocation`, `HierarchicalAxisCollocation` | axis nodes **and** implied logical evaluations | Separable models and nested axis-aligned structure |

`RARD` is retained for oscillatory or distributed residual structure when the
candidate and training budgets are large enough to resolve the residual field.
Its inactive slots receive zero loss weight and are activated incrementally from
the residual-weighted candidate distribution. Use an independent monitor: the
method is not a low-budget default.

For a fixed-capacity axis hierarchy, use `NestedDyadicAxisSpec(...)` with
`initial_level=...` on every adaptive axis and select
`phx.sampling.collocation.HierarchicalAxisCollocation(...)`. Inactive nodes remain
in the static JAX shape but receive zero active weight; refreshes activate nested
nodes without recompilation. Hierarchical-axis proposals must be validation-gated:
activation can improve solution-grid error while worsening the independently
measured PDE residual.

When `log_terms=True`, adaptive diagnostics are appended to each training term:
refresh counters, point/logical/axis-node counts, effective sample size, active
counts, and candidate evaluation counts. Compare methods at equal **residual
evaluation** and **logical evaluation** budgets; equal retained point counts alone
are not an equal-cost comparison. Controlled policies additionally report refresh
attempts/accepts/rejections, monitor mean/RMS/maximum, suspension state, and
cumulative candidate, monitor, and training residual-evaluation counts.

Set `profile_adaptive=True` on `solve(...)` only when measuring the eager refresh
boundary. It device-synchronizes refresh and optimizer work and stores
`refresh_wall_time_seconds` and `optimizer_wall_time_seconds` in
`trained_solver.training_diagnostics`. Leave it disabled for ordinary training,
because synchronization changes execution timing.

### `jit` and `keep_best`

- If `jit=True`, the per-step update is JIT-compiled when using standard Optax optimizers.
  (Line-search optimizers are not JIT-wrapped.)
- If `keep_best=True`, the returned solver uses the best parameter set observed over all epochs
  (by objective value); otherwise it returns the final parameters.
- `train_term_sample_size=k` samples `k` training terms per Optax step and
  rescales their losses to estimate the complete term sum. This is useful when
  terms have different static shapes, because JIT can compile smaller per-subset
  steps instead of one large graph containing every term.

### Logging and TensorBoard

`solve(...)` can report progress to stdout, a text file, TensorBoard, or any combination
of those outputs.

- `log_every`: console/file logging cadence. Use `0` to disable text progress logs.
- `log_terms`: include per-term and per-model-loss values in text logs and TensorBoard.
- `log_path`: write text logs to a file instead of stdout.
- `tensorboard_log_dir`: write TensorBoard event files.
- `tensorboard_every`: TensorBoard scalar cadence. By default it follows `log_every`
  when `log_every > 0`, otherwise it writes every iteration.

For data-fit terms such as `SupervisedDatasetTerm`, `RaggedTimeSeriesDataTerm`,
or `TrajectoryCaseDataTerm`, per-term logs also include supervised-data
diagnostics:

- `data_accuracy`: `1 - data_relative_l2_error`
- `data_relative_l2_error`: prediction-target relative L2 error
- `data_rmse`: root mean squared prediction-target error

When ragged trajectory data is enforced with
`phx.enforcement.enforce_ragged_time_series(...)`, the data is part of the ansatz
rather than a loss term. Train with physics penalties only, and put a
`RaggedTimeSeriesDataTerm` in `evaluation_terms` if diagnostics are required. Use
`interpolation="cubic_hermite"` when the physics residual needs second time
derivatives.

TensorBoard writes aggregate training scalars under `train/...`, training-term
scalars under `train/terms/...`, and evaluation-only term scalars under
`eval/terms/...`.

```text
solver = solver.solve(
    num_iter=200,
    optim=optax.adam(1e-3),
    seed=0,
    log_every=10,
    tensorboard_log_dir="runs/example",
    tensorboard_every=1,
)
```

View the run with:

```bash
tensorboard --logdir runs
```

## Training and evaluation data terms

Use index-scoped empirical terms for held-out validation. For ordinary row-wise
data, split the row indices and pass the training term through `terms` and the
validation term through `evaluation_terms`:

```python
import jax.numpy as jnp
import jax.random as jr
import phydrax as phx

rows = jnp.asarray([[0.0], [0.2], [0.4], [0.6], [0.8], [1.0]])
targets = 1.0 + 2.0 * rows[:, 0]
domain = phx.domain.DatasetDomain(rows)
train_idx, val_idx = phx.data_utils.train_test_split_indices(
    domain.size,
    test_fraction=0.2,
    key=jr.key(0),
)

train_data = phx.terms.SupervisedDatasetTerm(
    "u",
    domain.component(),
    targets,
    sampling=phx.domain.PointSampling(32, design="uniform"),
    indices=train_idx,
    label="train_data",
)
val_data = phx.terms.SupervisedDatasetTerm(
    "u",
    domain.component(),
    targets,
    sampling=phx.domain.PointSampling(32, design="uniform"),
    indices=val_idx,
    label="val_data",
)
```

For ragged trajectories, use `case_indices=...` on
`RaggedTimeSeriesDataTerm` or `TrajectoryCaseDataTerm`. The split is by dataset
row, so all observations from a held-out trajectory remain held out.

## Minimal example

```python
import equinox as eqx
import jax.random as jr
import optax
import phydrax as phx

geom = phx.domain.Interval1d(0.0, 1.0)

# Trainable scalar field u_theta(x)
model = phx.nn.models.MLP(
    in_size=1,
    out_size="scalar",
    width_size=16,
    depth=2,
    key=eqx.internal.doc_repr(jr.key(0), "jr.key(0)"),
)
u = geom.Model("x")(model)

layout = phx.domain.SampleLayout((("x",),))
component = geom.component()

# A toy residual that encourages u(x) ≈ 0 in Ω.
condition = phx.conditions.Residual("u", component, lambda f: f)
source = phx.integration.per_step(
    phx.integration.mean_over(component),
    phx.domain.PointSampling(128, layout=layout),
)
term = phx.terms.ResidualPenalty(condition, source)

solver = phx.solver.FunctionalSolver(functions={"u": u}, terms=[term])
loss0 = solver.loss(key=eqx.internal.doc_repr(jr.key(0), "jr.key(0)"))
solver = solver.solve(num_iter=20, optim=optax.adam(1e-3), seed=0)
loss1 = solver.loss(key=jr.key(1))
print(loss0, loss1)
```
