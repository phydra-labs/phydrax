# Solvers and training

This guide explains how `FunctionalSolver` evaluates losses and how `solve()` updates parameters.

## What `FunctionalSolver` does

A `FunctionalSolver` is a lightweight orchestrator that holds:

- `functions`: a mapping `{name: DomainFunction}` of the current fields,
- `constraints`: a list/tuple of constraint objects, each producing a scalar loss,
- `eval_constraints`: optional constraints used only for diagnostics/logging,
- optional `constraint_pipelines`: enforced-constraint pipelines that replace raw fields with ansatz
  functions satisfying selected conditions exactly.

The total objective is the sum of constraint losses:

$$
L = \sum_i \ell_i.
$$

`eval_constraints` are evaluated against the same current ansatz functions, but
they do not contribute to `loss(...)`, gradients, optimizer state, or best-model
selection. Use them for validation folds, held-out cases, or non-training
diagnostics.

## Loss evaluation (`loss(...)`)

When you call `solver.loss(key=...)`:

1) If enforced pipelines are configured, the current `functions` mapping is transformed into
   *ansatz functions* via `solver.ansatz_functions()`.
2) The provided PRNG key is split into one subkey per constraint.
3) Each constraint loss is evaluated and summed.

Additional keyword arguments are forwarded to each constraint's `.loss(...)` method.

## Enforced-constraint pipelines

Enforced pipelines are optional, but common when you want to enforce boundary/initial conditions
exactly by construction (rather than penalizing violations).

Pipelines are applied before any soft constraints are evaluated, so all residuals see the
post-processed (enforced) fields.

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

### Iteration counter (`iter_`)

During training, the current epoch index is passed to each constraint loss as `iter_` (as a JAX
scalar), so constraints can implement schedules (annealing, curriculum weights, etc.).

### `jit` and `keep_best`

- If `jit=True`, the per-step update is JIT-compiled when using standard Optax optimizers.
  (Line-search optimizers are not JIT-wrapped.)
- If `keep_best=True`, the returned solver uses the best parameter set observed over all epochs
  (by objective value); otherwise it returns the final parameters.

### Logging and TensorBoard

`solve(...)` can report progress to stdout, a text file, TensorBoard, or any combination
of those outputs.

- `log_every`: console/file logging cadence. Use `0` to disable text progress logs.
- `log_constraints`: include per-constraint losses in text logs and TensorBoard.
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
