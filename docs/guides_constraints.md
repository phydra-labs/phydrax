# Constraints and objectives

This guide distinguishes residual/data constraints from raw scalar objectives. Both
implement `.loss(functions, key=..., ...)` and can be summed by `FunctionalSolver`,
but they encode different mathematics.

## Constraint terms

A constraint is a scalar penalty $\ell(\theta)$ computed from domain-aware fields.
Sampled residual constraints apply squared-Frobenius semantics, so they are
nonnegative and express residual minimization or data fitting.

## Raw objective terms

A raw objective $\mathcal F(\theta)$ is added to the training functional without
squaring or forcing nonnegativity. `phydrax.objectives.IntegralFunctional` evaluates
a signed integral

$$
\mathcal F[u]=w\int_{\Omega_{\mathrm{comp}}} f[u](z)\,d\mu(z).
$$

Use it for Ritz energies and total-potential minimization. Do not replace a residual
constraint with a raw integral unless the underlying variational principle calls for
that signed functional. A stationary action is not generally a minimization problem.
`IntegralFunctional` requires its integrand to evaluate to a real scalar. Complex
output is rejected rather than silently truncated; use `real_part(...)` only when
selecting the real part is part of the objective's definition.

## Sampled (continuous) constraints

Many constraints are defined by:

1) a **domain component** (interior, boundary, initial slice, etc.), and  
2) a **residual operator** producing a `DomainFunction` $r(z)$ from one or more fields.

The pointwise penalty is a Hermitian squared Frobenius norm:

$$
\rho(z) = \|r(z)\|_F^2 = \sum_i \overline{r_i(z)}\,r_i(z).
$$

This agrees with $\sum_i r_i^2$ for real residuals and remains real and nonnegative
for complex residuals such as Schrödinger equations.

Phydrax supports two reduction modes:

`reduction="mean"` (measure-normalized):

$$
\ell = w\,\frac{1}{\mu(\Omega_{\text{comp}})}\int_{\Omega_{\text{comp}}}\rho(z)\,\mathrm{d}\mu(z)
$$

`reduction="integral"` (unnormalized):

$$
\ell = w\int_{\Omega_{\text{comp}}}\rho(z)\,\mathrm{d}\mu(z)
$$

Here $\mu$ is the component measure induced by the domain (volume/area/length for interiors,
surface measure for boundaries, counting measure for fixed slices, etc.).

`weight` can be either:

- a scalar/array-like global multiplier $w$, or
- a `DomainFunction` used as a pointwise weight inside the reduction
  (i.e. $\rho(z)$ becomes $w(z)\rho(z)$ before mean/integral reduction).

### Sampling plans and `over=...`

Every sampled constraint receives one explicit plan:

- `PointSampling(count, layout=SampleLayout(...), design=...)` produces a
  `PointBatch`;
- `GridSampling(axes, dense=...)` produces a `GridBatch`; `dense` is a
  `PointSampling` plan for labels not placed on coordinate axes.

The `over` argument selects sampled axes to reduce:

- `over=None`: reduce every sampled axis;
- `over="x"`: reduce the block or coordinate axes owned by `"x"`;
- `over=("x", "t")`: reduce both named factors.

`sampling_mode="resample"` draws from the plan on every loss call.
`sampling_mode="fixed"` reuses a caller-supplied `fixed_batch`, or materializes
one from `fixed_batch_key` during construction. Fixed sampling is useful for
reproducible diagnostics or expensive axis grids.

### Filtering: `where` and `where_all`

Continuous constraints can restrict the sampling region via:

- `where={label: predicate}` (per-label filtering),
- `where_all=predicate` (global filtering on the full point tuple).

Conceptually this applies an indicator/mask inside the integral/mean.

## A common pattern: interior PDE residual

`ContinuousPointwiseInteriorConstraint` is a convenience wrapper for pointwise residual losses.

```python
import phydrax as phx

geom = phx.domain.Interval1d(0.0, 1.0)

@geom.Function("x")
def u(x):
    return x[0] ** 2

layout = phx.domain.SampleLayout((("x",),))
constraint = phx.constraints.ContinuousPointwiseInteriorConstraint(
    "u",
    geom,
    operator=lambda f: phx.operators.laplacian(f, var="x"),
    sampling=phx.domain.PointSampling(128, layout=layout),
    reduction="mean",
)
```

## Discrete and pointset constraints

For sensor/anchor data (discrete samples), Phydrax provides constraints that do not sample from a
component, but instead evaluate on explicit point sets (and typically reduce by mean/integral in
an analogous way).

For ragged trajectory data indexed by a `TrajectoryDatasetDomain`, use
`RaggedTimeSeriesDataConstraint`. It samples valid `(data, t)` pairs, compares the
model against the stored time-series values, and reports the same supervised metrics
as other data-fit constraints.

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
    sampling=phx.domain.PointSampling(
        16,
        layout=phx.domain.SampleLayout((("data", "t"),)),
        design="uniform",
    ),
    selection="observation_uniform",
)

loss = constraint.loss({"u": exact})
metrics = constraint.data_metrics({"u": exact})
```

Use `case_indices=...` to restrict a ragged data constraint to a train or
validation set of trajectory rows. This is case-level splitting: every observed
time point for an excluded case stays excluded.

When the ragged time series is an observed input/forcing rather than the learned
field itself, expose it as a fixed function with `TrajectorySignal`. Include the
fixed signal in the solver's `functions` mapping under the same name used by
`constraint_vars`.
It remains JAX-traceable numeric state but is not optimized by `solve(...)`.

```python
forcing = phx.constraints.TrajectorySignal(
    domain,
    values,
    interpolation="linear",
)

physics = phx.constraints.FunctionalConstraint.from_operator(
    component=domain.component(),
    operator=lambda u, s: phx.operators.partial_t(u, var="t")
    - phx.operators.partial_t(s, var="t"),
    constraint_vars=("u", "forcing"),
    sampling=phx.domain.PointSampling(
        128,
        layout=phx.domain.SampleLayout((("data", "t"),)),
    ),
)
```

For labels attached to the dataset row rather than to every time point, use
`TrajectoryCaseDataConstraint`:

```python
targets = jnp.asarray([[1.0, 0.0], [2.0, -1.0], [3.0, -2.0]])

case_data = phx.constraints.TrajectoryCaseDataConstraint(
    "theta",
    domain.component(),
    targets,
    sampling=phx.domain.PointSampling(32, design="uniform"),
)
```

For non-time empirical rows, use `DatasetDomain` with
`SupervisedDatasetConstraint`. The domain owns the row payloads, and the constraint
owns aligned targets:

```python
rows = jnp.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]])
dataset_targets = rows[:, 0] + 2.0 * rows[:, 1]
dataset_domain = phx.domain.DatasetDomain(rows)

@dataset_domain.Function("data")
def u(row):
    return row[0] + 2.0 * row[1]

data = phx.constraints.SupervisedDatasetConstraint(
    "u",
    dataset_domain.component(),
    dataset_targets,
    sampling=phx.domain.PointSampling(32, design="uniform"),
)
```

Use `indices=...` to train or evaluate on an explicit subset of dataset rows.
`phydrax.data_utils.train_test_split_indices(...)` and
`phydrax.data_utils.kfold_indices(...)` return index arrays for these arguments.

For exact row-wise enforcement, use a hard ansatz instead of a data loss:

```python
@domain.Function("data", "t")
def u_free(data, t):
    return data[0] + t

@domain.Function()
def rhs():
    return 1.0

u = phx.constraints.enforce_ragged_time_series(
    u_free,
    domain,
    values,
    interpolation="cubic_hermite",
    gate="sin4",
)

physics = phx.constraints.FunctionalConstraint.from_operator(
    component=domain.component(),
    operator=lambda u_fn: phx.operators.partial_t(u_fn, var="t") - rhs,
    constraint_vars="u",
    sampling=phx.domain.PointSampling(
        128,
        layout=phx.domain.SampleLayout((("data", "t"),)),
    ),
)
```

The hard ansatz matches every observed node by construction, so the data constraint
can be kept only for diagnostics. Use `interpolation="linear"` for first-order
physics, or `interpolation="cubic_hermite"` when second time derivatives appear in
the residual. `components=[0, 2]` enforces only selected trailing output components.

## Integral equality constraints

Integral constraints enforce targets of the form

$$
\int_{\Omega_{\text{comp}}} f(z)\,d\mu(z) = c,
$$

where the left-hand side is estimated via Monte Carlo or quadrature, depending on the batch.
See [Guides → Integrals and measures](guides_integrals.md) for the measure/weighting details.
