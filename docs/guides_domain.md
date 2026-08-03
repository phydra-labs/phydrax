# Domains and sampling

This guide explains Phydrax's *labeled domains* and the two sampling modes used throughout the
library: **paired point sampling** and **coord-separable grid sampling**.

## Labeled product domains

A Phydrax domain represents a product of labeled factors:

$$
\Omega = \Omega_{\ell_1}\times\cdots\times \Omega_{\ell_k},
$$

where each factor has a label like `"x"` (space) or `"t"` (time). Product domains are composed
with the `@` operator:

```python
import phydrax as phx

geom = phx.domain.Interval1d(0.0, 1.0)        # label "x"
time = phx.domain.TimeInterval(0.0, 2.0)    # label "t" (alias of ScalarInterval)
domain = geom @ time                        # labels ("x", "t")
```

For non-time scalar axes, use `ScalarInterval(start, end, label="...")`.

## Vector domains with HyperRectangle

Use `HyperRectangle` when the domain is an axis-aligned box in `R^d` and each
sample should remain one vector under a single label. This is the simplest shape
for feature vectors, parameter boxes, and tabular supervised learning data.

```python
import jax.numpy as jnp
import phydrax as phx

features = phx.domain.HyperRectangle(
    lower=jnp.zeros(6),
    upper=jnp.ones(6),
    label="x",
)

@features.Function("x")
def u(x):
    return jnp.sum(x)

points = jnp.array(
    [
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
    ]
)
values = jnp.sum(points, axis=1)

data = phx.constraints.DiscreteInteriorDataConstraint(
    "u",
    features,
    points=points,
    values=values,
)
```

`points` is the raw `(N, d)` array. Do not wrap it in a dictionary unless the
domain is a product with multiple labels. The function receives each row as one
`(d,)` vector named `"x"`.

## Empirical dataset domains

Use `DatasetDomain` when the row itself is an empirical case or condition. This is
the right shape when rows are PyTrees, multimodal inputs, branch conditions, or
finite cases that will later be paired with physical coordinates.

For row-aligned scalar/vector targets on the empirical rows themselves, use
`SupervisedDatasetConstraint`:

```python
import jax.numpy as jnp
import phydrax as phx

rows = jnp.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]])
targets = rows[:, 0] + 2.0 * rows[:, 1]
dataset_domain = phx.domain.DatasetDomain(rows)

@dataset_domain.Function("data")
def u(row):
    return row[0] + 2.0 * row[1]

constraint = phx.constraints.SupervisedDatasetConstraint(
    "u",
    dataset_domain.component(),
    targets,
    num_cases=32,
)
```

Use `HyperRectangle` when the feature dimensions are continuous variables of the
problem. Use `DatasetDomain` when the empirical row distribution is the domain you
want to average over.

## Ragged trajectory dataset domains

Use `TrajectoryDatasetDomain` when each dataset element represents a function,
forcing, parameter vector, or latent descriptor and has an associated time series
with a shared `dt` but a row-specific length.
Use `IrregularTrajectoryDatasetDomain` for the same paired case-time model when
each row carries explicit observation times and the time spacing is non-uniform.

This is different from `DatasetDomain(...) @ TimeInterval(...)`: a plain product
domain creates a rectangular product and does not know that each dataset row has
its own valid time grid. Trajectory dataset domains keep the `data` and `t` labels
paired, so sampling and fixed-end slices are row-aware.

Because the row and time are coupled, use `ProductStructure((("data", "t"),))`
for trajectory sampling, including fixed-time components.
The same paired batches support hard branch-conditional data enforcement through
`enforce_ragged_time_series`; choose `cubic_hermite` interpolation when second time
derivatives are part of the physics residual.

Keep static case features and time-varying signals separate. Store static scalars
or vectors in the `inputs` rows, and expose observed ragged signals with
`TrajectorySignal` when residuals need them. Per-case scalar/vector labels should
use `TrajectoryCaseDataConstraint` rather than being repeated as constant time
series.

```python
import jax.numpy as jnp
import jax.random as jr
import phydrax as phx

inputs = jnp.asarray([[0.0], [1.0], [2.0]])
lengths = jnp.asarray([2, 4, 3])
trajectory_domain = phx.domain.TrajectoryDatasetDomain(inputs, lengths, dt=0.5)

component = trajectory_domain.component()
structure = phx.domain.ProductStructure((("data", "t"),))
batch = component.sample(8, structure=structure, key=jr.key(0))

@trajectory_domain.Function("data", "t")
def u(data, t):
    return data[0] + t

values = u(batch)
```

```python
signal_values = (
    inputs[:, 0, None]
    + trajectory_domain.dt * jnp.arange(trajectory_domain.max_length)[None, :]
)
forcing = phx.constraints.TrajectorySignal(
    trajectory_domain,
    signal_values,
    interpolation="linear",
)

targets = jnp.asarray([[1.0, 0.0], [2.0, -1.0], [3.0, -2.0]])
case_target = phx.constraints.TrajectoryCaseDataConstraint(
    "theta",
    trajectory_domain.component(),
    targets,
    num_cases=32,
)
```

Functions on a domain are wrapped as `DomainFunction`s. The key idea is that a `DomainFunction`
declares which labels it depends on, and operators/constraints use those labels consistently.

```python
@domain.Function("x", "t")
def u(x, t):
    return x[0] * (1.0 + t)
```

## Components: interior, boundary, and fixed slices

Constraints and integrals are typically evaluated over a **domain component**, which selects a
subset of each factor:

- `Interior()`: the interior of a geometry or scalar interval;
- `Boundary()`: the boundary of a geometry or scalar interval (endpoints in 1D);
- `FixedStart()` / `FixedEnd()`: the start/end slice of a scalar interval (often time);
- `Fixed(value)`: a slice at a specified coordinate.

Components are created with `domain.component(...)`:

```python
# Continuing from: domain = geom @ time
component = domain.component({"t": phx.domain.FixedStart()})  # initial-time slice
```

A product domain's `boundary()` method returns a `DomainComponentUnion`: one
additive term for each codimension-one face. This collection models a
measure-disjoint decomposition, so all terms share the same domain and exact
duplicates are rejected. It is not a geometric Boolean union; overlapping
filtered terms would be counted once per term.

### Filtering with `where` and `where_all`

Sampling can be restricted by predicates:

- `where={label: predicate}` applies a per-label predicate, e.g. `where={"x": lambda x: x[0] < 0.5}`.
- `where_all=predicate` applies a predicate to the *full point tuple* (useful for coupled filters).

These filters behave like indicator functions: points that fail the predicate are discarded (for
point sampling) or masked out (for coord-separable sampling).

## Paired point sampling (`PointsBatch`)

Most pointwise PDE residual constraints use **paired sampling**, driven by a `ProductStructure`.

A `ProductStructure` partitions the sampled labels into blocks. Each block is sampled jointly,
and each block corresponds to one named sampling axis in the resulting `PointsBatch`.

Examples:

- `ProductStructure((("x", "t"),))` samples paired space-time points.
- `ProductStructure((("x",), ("t",)))` samples space and time independently (Cartesian product).

```python
import equinox as eqx
import jax.random as jr
import phydrax as phx

# Continuing from: domain = geom @ time
structure = phx.domain.ProductStructure((("x", "t"),))
batch = domain.component().sample(
    128,
    structure=structure,
    key=eqx.internal.doc_repr(jr.key(0), "jr.key(0)"),
)
```

### Deterministic and scrambled low-discrepancy samplers

`sampler="halton"` and `sampler="sobol"` select the standard unscrambled
low-discrepancy sequences. Their points are independent of the supplied random
key or host seed. Use `"halton_scrambled"` or `"sobol_scrambled"` when randomized
scrambling is required; a fixed key or seed reproduces the same scrambled
sequence, while different keys or seeds produce different points. Host samplers
and JAX callback samplers follow the same naming and reproducibility contract.

## Coord-separable grid sampling (`CoordSeparableBatch`)

For spectral/basis operators and neural operators, it is often preferable to sample *1D axes*
and evaluate on the implied Cartesian grid. This is **coord-separable sampling**.

You choose which unary labels are coord-separable by passing a per-label spec, e.g.
`{"x": FourierAxisSpec(64)}` for a 1D periodic grid or `{"x": (64, 64)}` for a 2D grid.

```python
import equinox as eqx
import jax.random as jr
import phydrax as phx

geom = phx.domain.Interval1d(0.0, 1.0)
batch = geom.component().sample_coord_separable(
    {"x": phx.domain.FourierAxisSpec(64)},
    key=eqx.internal.doc_repr(jr.key(0), "jr.key(0)"),
)
```

When a label is coord-separable, the value passed into a `DomainFunction` for that label
is a **tuple of 1D coordinate arrays** (for scalar labels this tuple has length 1),
rather than a point cloud.

## Axis specs and quadrature metadata

Axis specs (`FourierAxisSpec`, `LegendreAxisSpec`, etc.) can attach an `AxisDiscretization` to the
batch, including:

- `nodes` (the axis coordinates),
- optional quadrature weights (for `integral`/`mean`),
- basis metadata used by `backend="basis"` differential operators.

This is how Phydrax keeps sampling, quadrature, and operator discretization consistent without
manual bookkeeping.

### Nested axes and CAD cut-cell weights

`NestedDyadicAxisSpec(capacity, initial_level=...)` materializes a fixed-capacity
axis with immutable level, parent-interval, active-node, and quadrature metadata.
It is intended for `HierarchicalAxisCollocation`: the array shape remains fixed
while nested nodes become active.

For an irregular geometry sampled on bounding-box axes, `coord_mask_by_label`
remains a boolean support mask. A numerical geometry correction is stored
separately in `coord_geometry_weight_by_label`; integrals multiply both. Request a
deterministic subcell estimate with `GridSpec(..., cut_cell_order=k)`:

```python
cad = phx.domain.Circle(center=(0.0, 0.0), radius=1.0)
grid = phx.domain.GridSpec(
    (
        phx.domain.UniformAxisSpec(25),
        phx.domain.UniformAxisSpec(25),
    ),
    cut_cell_order=3,
)
batch = cad.component().sample_coord_separable({"x": grid})
```

Each tensor node represents its bounding Voronoi subcell. Phydrax probes each
subcell, estimates its occupied fraction, keeps evaluation points inside the
declared geometry, and normalizes the represented mass to the geometry measure.
This improves constant/low-order integration over a binary nodal mask. It does
not recover a disconnected feature that has no interior tensor node. Use a
paired-point representation or a geometry-conforming mesh for such features.
`CoordSeparableBatch` rejects negative, non-finite, misaligned, and unknown-label
geometry weights.

### CAD measure partitions and boundary charts

Mesh-backed 2D CAD geometries expose:

- `interior_measure_partition`: triangle areas;
- `boundary_measure_partition`: boundary-edge lengths;
- `boundary_chart_atlas`: one unit-interval chart per boundary edge.

Mesh-backed 3D CAD geometries expose a surface-triangle
`boundary_measure_partition` and `boundary_chart_atlas`. A 3D interior
volume-cell partition is intentionally not synthesized from a surface mesh.

`CADChartAtlas.tensor_quadrature(order)` maps Gauss-Legendre reference axes to
physical edges or Duffy-mapped triangles. `CADChartQuadrature.weights` already
contains the physical Jacobian and trim semantics. Adjacent charts share only
measure-zero seams, so summing chart weights does not double-count physical
surface measure:

```py
import jax.numpy as jnp

chart_rule = cad.boundary_chart_atlas.tensor_quadrature(6)
surface_measure = chart_rule.integrate(
    jnp.ones(chart_rule.weights.shape)
)
```

## Phase-space product domains (position–momentum)

You can represent phase space by composing a spatial geometry for position with a second spatial
geometry that you *relabel* as momentum.

### Without time

```python
import phydrax as phx

x = phx.domain.Interval1d(0.0, 1.0)              # label "x"
p = phx.domain.Interval1d(-2.0, 2.0).relabel("p")  # momentum axis, label "p"

phase = x @ p                                  # labels ("x", "p")

@phase.Function("x", "p")
def f(x, p):
    return x[0] ** 2 + p[0] ** 2

structure = phx.domain.ProductStructure((("x", "p"),))  # paired (x,p) samples
batch = phase.component().sample(256, structure=structure)
val = f(batch)  # evaluated on phase-space points
```

### With time

Add a time factor and treat the objective as evolving on $\Omega_x\times\Omega_p\times[t_0,t_1]$:

```python
import phydrax as phx

x = phx.domain.Interval1d(0.0, 1.0)
p = phx.domain.Interval1d(-2.0, 2.0).relabel("p")
t = phx.domain.TimeInterval(0.0, 5.0)          # label "t"

phase_time = x @ p @ t                         # labels ("x", "p", "t")

@phase_time.Function("x", "p", "t")
def f(x, p, t):
    return (x[0] ** 2 + p[0] ** 2) * (1.0 + t)

structure = phx.domain.ProductStructure((("x", "p", "t"),))
batch = phase_time.component().sample(512, structure=structure)
val = f(batch)
```

### Higher-dimensional momentum domains

For multi-dimensional momentum, relabel a 2D/3D geometry:

```python
import phydrax as phx

x = phx.domain.Square(center=(0.0, 0.0), side=2.0)            # "x" in R^2
p = phx.domain.Square(center=(0.0, 0.0), side=6.0).relabel("p")  # "p" in R^2
phase = x @ p
```
