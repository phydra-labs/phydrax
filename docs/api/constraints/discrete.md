# Discrete constraints

For `PointSetConstraint`, runtime operator knobs can be set once via
`eval_kwargs` and are merged into each `.loss(...)` call.

`PointSetConstraint.weight` can be a scalar global multiplier or a
`DomainFunction` evaluated pointwise on the anchor set.

Data-fit constraints created by `DiscreteInteriorDataConstraint`,
`DiscreteTimeDataConstraint`, and `SupervisedDatasetConstraint` attach
supervised-data diagnostics used by `FunctionalSolver.solve(...)` logging.
`RaggedTimeSeriesDataConstraint` provides the same diagnostics for
trajectory-valued targets, and `TrajectoryCaseDataConstraint` provides them for
per-row scalar/vector targets on a `TrajectoryDatasetDomain`.
`RaggedSeriesSupervisedConstraint` provides the same row-aligned diagnostics for
`RaggedSeriesDatasetDomain` inputs:

- `data_accuracy`
- `data_relative_l2_error`
- `data_rmse`

## Discrete point constraints

::: phydrax.constraints.PointSetConstraint
    options:
        members:
            - __init__
            - from_points
            - from_operator
            - data_metrics
            - loss

---

::: phydrax.constraints.DiscreteInteriorDataConstraint
    options:
        members: []

---

::: phydrax.constraints.DiscreteTimeDataConstraint
    options:
        members: []

## Supervised dataset constraints

Use `SupervisedDatasetConstraint` when a `DatasetDomain` row is the sampled
empirical case and the target is aligned by row index. This is the finite-case
counterpart to `DiscreteInteriorDataConstraint`: the latter treats observed inputs
as points in a continuous coordinate domain, while this treats the empirical row
measure itself as the domain.

```python
import jax.numpy as jnp
import phydrax as phx

rows = jnp.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]])
targets = rows[:, 0] + 2.0 * rows[:, 1]
domain = phx.domain.DatasetDomain(rows)

@domain.Function("data")
def exact(row):
    return row[0] + 2.0 * row[1]

constraint = phx.constraints.SupervisedDatasetConstraint(
    "u",
    domain.component(),
    targets,
    num_cases=16,
)

loss = constraint.loss({"u": exact})
metrics = constraint.data_metrics({"u": exact})
```

Pass `indices=...` to restrict sampling to a train/validation subset of dataset
rows. This is the recommended way to pair `SupervisedDatasetConstraint` with
`FunctionalSolver(eval_constraints=...)`.

::: phydrax.constraints.SupervisedDatasetConstraint
    options:
        members:
            - __init__
            - sample
            - data_metrics
            - loss

---

::: phydrax.constraints.SupervisedDatasetBatch
    options:
        members: []

## Supervised ragged series constraints

Use `RaggedSeriesSupervisedConstraint` when the input row contains padded
variable-length series plus a mask and the target is aligned by case. This is the
right shape for conditional mappings such as `(static, four sensor traces) -> two
summary scalars`.

```python
import jax.numpy as jnp
import phydrax as phx

static = jnp.asarray([[1.0, 0.0], [2.0, 1.0]])
series = jnp.asarray(
    [
        [[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]],
        [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
    ]
)
lengths = jnp.asarray([2, 3])
targets = jnp.asarray([[4.0, -4.0], [21.0, -21.0]])

domain = phx.domain.RaggedSeriesDatasetDomain(series, lengths, static=static)

def encoder(payload, *, key=None):
    del key
    valid = payload.mask.astype(float)
    total = jnp.sum(payload.series[..., 0] * valid, axis=1)
    return jnp.stack((total, -total), axis=-1)

u = domain.Function("data")(phx.nn.RaggedSeriesModel(encoder))
constraint = phx.constraints.RaggedSeriesSupervisedConstraint(
    "u",
    domain.component(),
    targets,
    num_cases=16,
)
```

Pass `indices=...` for train/validation splits at the case level.

For long series, avoid full padded rows during training by sampling fixed-width
views:

```python
constraint = phx.constraints.RaggedSeriesSupervisedConstraint(
    "u",
    domain.component(),
    targets,
    num_cases=64,
    series_sampling="points_uniform",
    num_series_points=256,
)
```

The default `series_sampling="full"` preserves the full-row behavior. Sampled
modes return `(num_cases, num_series_points, ...)` series payloads and keep the
same per-case targets. Use `window_uniform` when local order matters, and
`points_uniform` for permutation-invariant pooling/statistical encoders.

For full-sequence training on long, uneven records, prefer length buckets over
global full rows:

```python
import optax

constraints = phx.constraints.RaggedSeriesSupervisedConstraint.bucketed(
    "u",
    domain.component(),
    targets,
    num_cases=64,
    num_buckets=8,
)
solver = phx.solver.FunctionalSolver(functions={"u": u}, constraints=constraints)
solver = solver.solve(
    num_iter=1000,
    optim=optax.rprop(1e-3),
    train_constraint_sample_size=1,
)
```

The helper returns multiple ordinary constraints. Each bucket uses a fixed-width
prefix view whose width is the bucket maximum length, so it still covers the
complete series for every case in that bucket. Bucketing is only an efficiency
device: `num_cases` is distributed across buckets by bucket population, and
bucket losses are scaled so the combined estimator matches one full padded
constraint with the same reduction. Use `train_constraint_sample_size=1` when
there are many bucket shapes; this compiles one bucket-shaped step at a time
instead of one large graph containing every bucket.

::: phydrax.constraints.RaggedSeriesSupervisedConstraint
    options:
        members:
            - __init__
            - bucketed
            - sample
            - data_metrics
            - loss

---

::: phydrax.constraints.RaggedSeriesSupervisedBatch
    options:
        members: []

## Supervised graph constraints

Use `GraphTarget` when fixed graph-family data should be consumed as a
`DomainFunction` inside a physics residual. Use `GraphSupervisedConstraint` when
the graph target is the supervised residual itself. Targets are supplied per graph
case and are aligned to sampled node, edge, or global rows.

`GraphTrajectorySignal` and `GraphTrajectorySupervisedConstraint` provide the
same pattern for graph trajectories with values shaped by case, time, and graph
entity.

::: phydrax.constraints.GraphTarget
    options:
        members: []

---

::: phydrax.constraints.GraphSupervisedConstraint
    options:
        members: []

---

::: phydrax.constraints.GraphTrajectorySignal
    options:
        members: []

---

::: phydrax.constraints.GraphTrajectorySupervisedConstraint
    options:
        members: []

## Ragged trajectory constraints

Use `RaggedTimeSeriesDataConstraint` when each dataset row has a vector-valued time
series with the same `dt` but a row-specific length.

```python
import jax.numpy as jnp
import phydrax as phx

inputs = jnp.asarray([[0.0], [1.0], [2.0]])
lengths = jnp.asarray([2, 4, 3])
domain = phx.domain.TrajectoryDatasetDomain(inputs, lengths, dt=0.5)

times = domain.start + domain.dt * jnp.arange(domain.max_length)
values = inputs[:, 0, None] + times[None, :]

@domain.Function("data", "t")
def exact(data, t):
    return data[0] + t

constraint = phx.constraints.RaggedTimeSeriesDataConstraint(
    "u",
    domain.component(),
    values,
    num_points=16,
)
loss = constraint.loss({"u": exact})
metrics = constraint.data_metrics({"u": exact})
```

Pass `case_indices=...` to restrict sampling to selected trajectory rows. For
ragged train/test splits this keeps every observed time point from a held-out
trajectory out of the training data constraint.

::: phydrax.constraints.RaggedTimeSeriesDataConstraint
    options:
        members:
            - __init__
            - sample
            - data_metrics
            - loss

---

::: phydrax.constraints.RaggedTimeSeriesBatch
    options:
        members: []

## Fixed trajectory signals and case targets

Use `TrajectorySignal` when ragged time-series data is an observed input/forcing
rather than a supervised output. It returns a fixed `DomainFunction` over
`(data, t)`, so physics residuals can consume it alongside learned fields. Include
the fixed signal in the solver's `functions` mapping under the same name used by
the residual constraint.

Use `TrajectoryCaseDataConstraint` when the target belongs to the dataset row, not
to every time point. This is the right shape for scalar/vector labels such as
parameters, final summaries, or class logits.

```python
static = jnp.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]])
lengths = jnp.asarray([2, 4, 3])
domain = phx.domain.TrajectoryDatasetDomain(static, lengths, dt=0.5)

times = domain.start + domain.dt * jnp.arange(domain.max_length)
forcing_values = static[:, 0, None] + times[None, :]
forcing = phx.constraints.TrajectorySignal(
    domain,
    forcing_values,
    interpolation="linear",
)

targets = jnp.stack((static[:, 0] + static[:, 1], static[:, 0] - static[:, 1]), axis=-1)
case_data = phx.constraints.TrajectoryCaseDataConstraint(
    "theta",
    domain.component(),
    targets,
    num_cases=32,
)

physics = phx.constraints.FunctionalConstraint.from_operator(
    component=domain.component(),
    operator=lambda u, s: phx.operators.partial_t(u, var="t")
    - phx.operators.partial_t(s, var="t"),
    constraint_vars=("u", "forcing"),
    num_points=128,
    structure=phx.domain.ProductStructure((("data", "t"),)),
)
```

`TrajectorySignal(interpolation="linear")` supports first time derivatives.
`interpolation="cubic_hermite"` supports first and second time derivatives.
`interpolation="nearest"` is for value lookup only. Trajectory signals are fixed
numeric state: they can be traced through JAX residuals but are excluded from
solver optimizer parameters.
`TrajectoryCaseDataConstraint` also accepts `case_indices=...` for row-level
train/eval splits.

::: phydrax.constraints.TrajectorySignal
    options:
        members: []

---

::: phydrax.constraints.TrajectoryCaseDataConstraint
    options:
        members:
            - __init__
            - sample
            - data_metrics
            - loss

---

::: phydrax.constraints.TrajectoryCaseDataBatch
    options:
        members: []

## Hard ragged trajectory enforcement

`enforce_ragged_time_series` converts a free model into a hard ansatz that exactly
matches every observed ragged trajectory node. Use this when the data should define
the solution manifold and the solver should train only physics residuals.

```python
@domain.Function("data", "t")
def u_free(data, t):
    return data[0] + t

u = phx.constraints.enforce_ragged_time_series(
    u_free,
    domain,
    forcing_values,
    interpolation="cubic_hermite",
    gate="sin4",
)
```

`interpolation="linear"` supports value enforcement and first time derivatives.
`interpolation="cubic_hermite"` supports first and second time derivatives using
finite-difference node slopes. `gate="sin2"` is compact and smooth for first-order
physics; `gate="sin4"` keeps the gate flatter at observed nodes and is the better
default for second-order time residuals. Pass `components=[...]` to hard-enforce
only selected trailing output components while leaving the others free.

::: phydrax.constraints.enforce_ragged_time_series
    options:
        members: []

## Discrete boundary / initial constraints

::: phydrax.constraints.DiscreteDirichletBoundaryConstraint
    options:
        members: []

---

::: phydrax.constraints.DiscreteNeumannBoundaryConstraint
    options:
        members: []

---

::: phydrax.constraints.DiscreteInitialConstraint
    options:
        members: []

## Discrete ODE constraints

::: phydrax.constraints.DiscreteODEConstraint
    options:
        members: []

## Probabilistic supervised constraints

`SupervisedLikelihoodConstraint` shares `DatasetDomain` row-index semantics with
`SupervisedDatasetConstraint`, but minimizes negative log likelihood and can apply a
physical observation operator before scoring.

::: phydrax.constraints.SupervisedLikelihoodConstraint
    options:
        members:
            - __init__
            - sample
            - loss
