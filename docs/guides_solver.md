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
- `constraints`: a list/tuple of constraint objects, each producing a scalar loss,
- `objectives`: raw scalar terms such as signed integral energies,
- model-level losses attached to models with `model.add_model_loss(...)` or a custom
  model `__loss__` hook,
- `eval_constraints`: optional constraints used only for diagnostics/logging,
- optional `constraint_pipelines`: enforced-constraint pipelines that replace raw fields with ansatz
  functions satisfying selected conditions exactly.

The training functional is the sum of constraint losses, raw objective terms, and
attached model losses:

$$
\mathcal J = \sum_i \ell_i + \sum_j \mathcal F_j + \sum_k r_k.
$$

`eval_constraints` are evaluated against the same current ansatz functions, but
they do not contribute to `loss(...)`, gradients, optimizer state, or best-model
selection. Use them for validation folds, held-out cases, or non-training
diagnostics.

## Loss evaluation (`loss(...)`)

When you call `solver.loss(key=...)`:

1) If enforced pipelines are configured, the current `functions` mapping is transformed into
   *ansatz functions* via `solver.ansatz_functions()`.
2) The provided PRNG key is split into independent subkeys for constraints, objectives,
   and model losses.
3) Each constraint loss and raw objective term is evaluated and summed.
4) Model-level losses attached to the raw trainable models are evaluated and added.

Additional keyword arguments are forwarded to each constraint and objective `.loss(...)`.
The `iter_` keyword, when present, is also forwarded to model losses.

Raw objective terms may be negative. `IntegralFunctional` integrates its density
without squaring it; use this for energy/Ritz minimization, not for a residual penalty.

### Sampled objective materialization

Objectives derived from `AbstractSamplingObjectiveTerm` own two stages:

1. `sample(key=...)` materializes an immutable batch of paths, labels, particles, or
   derivative probes;
2. `loss(..., batch=batch)` evaluates the current model on that batch.

For Optax training, `FunctionalSolver` performs stage 1 once per optimizer update,
outside `filter_value_and_grad` and the compiled loss. The identical batch is reused
for every value, gradient, line-search, and same-update parameter-view evaluation.
This prevents differentiation through Monte Carlo target construction and prevents
an optimizer population from silently receiving different targets within one update.
Evosax likewise materializes one batch per population update.

Fixed sampled objectives return the same batch on every call and are the default for
common-random-number comparisons. Resampled objectives receive deterministic fresh
subkeys each update. Multiple sampled objectives receive distinct subkeys. Keep probe
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

model = phx.nn.MLP(
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
best-model selection. When `log_constraints=True`, text logs print them as
`[model i] ...`, and TensorBoard writes them under `train/model_losses/...`.

## Enforced-constraint pipelines

Enforced pipelines are optional, but common when you want to enforce boundary/initial conditions
exactly by construction (rather than penalizing violations).

Pipelines are applied before any soft constraints are evaluated, so all residuals see the
post-processed (enforced) fields.

Geometry Dirichlet ansätze and pipeline preservation overlays use a dimensionless
gate that is broad by default. CAD Neumann, Robin, Sommerfeld, and traction ansätze
use a dimensional rescaling of the same global R-equivalence profile whose outward
normal derivative is one. Their interior residuals use the factor gradient as a
smooth normal extension; it equals the outward unit normal at regular boundary
points but can vanish at medial sets. Public normal calculations continue to use the
geometry ADF and normal provider. Select `gate_method="compact"` to use the compact
Dirichlet and overlay fallback; only then do `gate_saturation_fraction` and
`gate_linear_fraction` tune its transition. Exact analytic geometry gates ignore
these mesh-specific controls.

Boundary constraints whose operators contain spatial derivatives must declare
that order on `SingleFieldEnforcedConstraint` or
`MultiFieldEnforcedConstraint`. Set `max_derivative_order=0` for Dirichlet
values and `max_derivative_order=1` for Neumann, Robin, Sommerfeld, or traction
conditions. Later initial/data overlays are then multiplied by
$\beta^{K+1}$, preserving boundary derivatives through order $K$.

See [API → Solver → Enforced constraint pipelines](api/solver/enforced_constraints.md) for the pipeline
types and constructors.

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
- an Optax `GradientTransformationExtraArgs` (line-search style optimizers),
- an evosax algorithm instance (evolutionary strategies).

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

During training, the current epoch index is passed to each constraint loss as `iter_` (as a JAX
scalar), so constraints can implement schedules (annealing, curriculum weights, etc.).

### Adaptive collocation policies

Attach an adaptive policy with `collocation_policy=...` when constructing a
`FunctionalConstraint`. `FunctionalSolver` initializes one immutable population per
adaptive constraint, refreshes it before eligible optimizer steps, passes the explicit
batch and estimator weight into the loss, and returns the updated population on the
trained solver. Calling `solve(...)` again continues from that population.

The unconditional default remains fixed scrambled Sobol sampling:

```python
sampling_mode = "fixed"
sampler = "sobol_scrambled"
collocation_policy = None
```

Use `phx.constraints.RECOMMENDED_COLLOCATION_DEFAULTS` to inspect that contract.
When adaptive refinement is explicitly requested, R3 is the supported general
starting point. Support is declared by `COLLOCATION_POLICY_SUPPORT` and
`collocation_policy_support(...)`:

- **Stable:** fixed scrambled Sobol, periodic replacement, R3, and periodic
  separable sampling.
- **Conditional:** RAR-D for sufficiently resolved oscillatory residuals and
  hierarchical-axis refinement for nested coordinate-separable discretizations.

For production-style adaptive runs, separate proposal generation from solver
control with `controlled_collocation(...)`:

```python
policy = phx.constraints.controlled_collocation(
    phx.constraints.R3(
        refresh_every=25,
        sampler="sobol_scrambled",
        min_replace_fraction=0.1,
        max_retain_fraction=0.9,
    ),
    schedule=phx.constraints.RefreshSchedule(25),
    monitor=phx.constraints.ResidualMonitor(sampler="sobol_scrambled"),
    guard=phx.constraints.RefreshGuard(
        max_relative_regression=0.0,
        max_consecutive_rejections=2,
        suspension_steps=100,
    ),
    budget=phx.constraints.AdaptationBudget(
        max_candidate_evaluations=100_000,
        max_monitor_evaluations=25_000,
    ),
    anchors=phx.constraints.CoverageAnchors(0.25),
)
```

The controller maintains three distinct populations:

1. the mutable training population proposed by the wrapped policy,
2. a fixed independent monitor population used for acceptance and rollback,
3. an untouched test population owned by the application, never by the solver.

A proposed population is evaluated on the next eligible control step. Monitor
regression rejects and rolls it back; repeated rejection suspends adaptation for
the configured cooldown. `FunctionalSolver.solve(...)` explicitly settles any
proposal still pending after the final optimizer step, without admitting another
proposal. Coverage anchors reserve a persistent low-discrepancy fraction of paired
populations so residual concentration cannot consume all global coverage. Monitor
budgets reserve the next validation evaluation before a proposal is admitted, so
exhausting a candidate or monitor budget cannot leave the final population
permanently unvalidated.

Choose the population representation first:

| Representation | Policies | Budget unit | Appropriate support |
| --- | --- | --- | --- |
| Paired `PointsBatch` | `PeriodicCollocation`, `R3`, `RARD` | retained points and residual candidate evaluations | General pointwise PINNs |
| Coordinate-separable tensor | `PeriodicSeparableCollocation`, `HierarchicalAxisCollocation` | axis nodes **and** implied logical evaluations | Separable models and nested axis-aligned structure |

`RARD` is retained for oscillatory or distributed residual structure when the
candidate and training budgets are large enough to resolve the residual field.
Its inactive slots receive zero loss weight and are activated incrementally from
the residual-weighted candidate distribution. Use an independent monitor: the
method is not a low-budget default.

For a coordinate-separable fixed-capacity hierarchy, use
`NestedDyadicAxisSpec(..., initial_level=...)` on every adaptive axis and attach
`HierarchicalAxisCollocation(...)`. Inactive nodes remain in the static JAX shape but
receive zero active weight; refreshes activate nested nodes without recompilation.

Hierarchical-axis proposals must be validation-gated. They can substantially
improve solution-grid error while worsening the independently measured PDE
residual, so activation alone is not sufficient evidence of improvement.

When `log_constraints=True`, adaptive diagnostics are appended to each training
term: refresh counters, point/logical/axis-node counts, effective sample size,
active counts, and candidate evaluation counts. Compare methods at equal
**residual evaluation** and **logical evaluation** budgets; equal retained point
counts alone are not an equal-cost comparison. Controlled policies additionally
report refresh attempts/accepts/rejections, monitor mean/RMS/maximum, suspension
state, and cumulative candidate, monitor, and training residual-evaluation counts.

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
- `train_constraint_sample_size=k` samples `k` training constraints per Optax step
  and rescales their losses to estimate the full constraint sum. This is useful
  when many constraints have different static shapes, because JIT can compile
  smaller per-subset steps instead of one large graph containing every
  constraint.

### Logging and TensorBoard

`solve(...)` can report progress to stdout, a text file, TensorBoard, or any combination
of those outputs.

- `log_every`: console/file logging cadence. Use `0` to disable text progress logs.
- `log_constraints`: include per-constraint and per-model-loss terms in text logs and TensorBoard.
- `log_path`: write text logs to a file instead of stdout.
- `tensorboard_log_dir`: write TensorBoard event files.
- `tensorboard_every`: TensorBoard scalar cadence. By default it follows `log_every`
  when `log_every > 0`, otherwise it writes every iteration.

For data-fit constraints created by `DiscreteInteriorDataConstraint`,
`DiscreteTimeDataConstraint`, `SupervisedDatasetConstraint`,
`RaggedTimeSeriesDataConstraint`, or `TrajectoryCaseDataConstraint`,
per-constraint logs also include supervised-data diagnostics:

- `data_accuracy`: `1 - data_relative_l2_error`
- `data_relative_l2_error`: prediction-target relative L2 error
- `data_rmse`: root mean squared prediction-target error

When ragged trajectory data is enforced with
`enforce_ragged_time_series(...)`, the data is part of the ansatz rather than a
loss term. In that case, train with physics constraints only and keep a
`RaggedTimeSeriesDataConstraint` outside the solver if you want diagnostics. Use
`interpolation="cubic_hermite"` when the physics residual needs second time
derivatives.

TensorBoard writes aggregate training scalars under `train/...`, training
constraint scalars under `train/constraints/...`, and eval-only constraint scalars
under `eval/constraints/...`. The legacy `constraints/...` train tags are also
emitted for existing dashboards.

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

## Train/eval empirical constraints

Use index-scoped empirical constraints for held-out validation. For ordinary
row-wise data, split the row indices and give training indices to `constraints`
and validation indices to `eval_constraints`:

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

train_data = phx.constraints.SupervisedDatasetConstraint(
    "u",
    domain.component(),
    targets,
    num_cases=32,
    indices=train_idx,
    label="train_data",
)
val_data = phx.constraints.SupervisedDatasetConstraint(
    "u",
    domain.component(),
    targets,
    num_cases=32,
    indices=val_idx,
    label="val_data",
)
```

For ragged trajectories, use `case_indices=...` on
`RaggedTimeSeriesDataConstraint` or `TrajectoryCaseDataConstraint`. The split is
by dataset row, so all observations from a held-out trajectory remain held out.

## Minimal example

```python
import equinox as eqx
import jax.random as jr
import optax
import phydrax as phx

geom = phx.domain.Interval1d(0.0, 1.0)

# Trainable scalar field u_theta(x)
model = phx.nn.MLP(
    in_size=1,
    out_size="scalar",
    width_size=16,
    depth=2,
    key=eqx.internal.doc_repr(jr.key(0), "jr.key(0)"),
)
u = geom.Model("x")(model)

structure = phx.domain.ProductStructure((("x",),))

# A toy interior objective that encourages u(x) ≈ 0 in Ω (replace with a PDE operator in real use).
constraint = phx.constraints.ContinuousPointwiseInteriorConstraint(
    "u",
    geom,
    operator=lambda f: f,
    num_points=128,
    structure=structure,
    reduction="mean",
)

solver = phx.solver.FunctionalSolver(functions={"u": u}, constraints=[constraint])
loss0 = solver.loss(key=eqx.internal.doc_repr(jr.key(0), "jr.key(0)"))
solver = solver.solve(num_iter=20, optim=optax.adam(1e-3), seed=0)
loss1 = solver.loss(key=jr.key(1))
print(loss0, loss1)
```

## EvoSax example (gradient-free)

To use evolutionary strategies, pass an evosax algorithm instance as `optim=...`:

```python
import equinox as eqx
from evosax import algorithms as evo_algos
import phydrax as phx

# Continuing from the minimal example above:
# solver = phx.solver.FunctionalSolver(functions={"u": u}, constraints=[constraint])

# evosax expects a "solution" PyTree matching the trainable parameter structure.
params = solver.trainable_functions()
algo = evo_algos.Open_ES(population_size=8, solution=params)

solver = solver.solve(num_iter=20, optim=algo, seed=0)
```
