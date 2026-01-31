# Solvers and training

This guide explains how `FunctionalSolver` evaluates losses and how `solve()` updates parameters.

## What `FunctionalSolver` does

A `FunctionalSolver` is a lightweight orchestrator that holds:

- `functions`: a mapping `{name: DomainFunction}` of the current fields,
- `constraints`: a list/tuple of constraint objects, each producing a scalar loss,
- optional `constraint_pipelines`: enforced-constraint pipelines that replace raw fields with ansatz
  functions satisfying selected conditions exactly.

The total objective is the sum of constraint losses:

$$
L = \sum_i \ell_i.
$$

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
`solver.functions`. Under the hood it uses Equinox to split the function PyTree into:

- **trainable parameters**: inexact arrays (floating/complex arrays),
- **static part**: everything else.

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
params, _ = eqx.partition(solver.functions, eqx.is_inexact_array)
algo = evo_algos.Open_ES(population_size=8, solution=params)

solver = solver.solve(num_iter=20, optim=algo, seed=0)
```
