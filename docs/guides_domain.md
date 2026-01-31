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

## Coord-separable grid sampling (`CoordSeparableBatch`)

For spectral/basis operators and neural operators, it is often preferable to sample *1D axes*
and evaluate on the implied Cartesian grid. This is **coord-separable sampling**.

You choose which geometry labels are coord-separable by passing a per-label spec, e.g.
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

When a geometry label is coord-separable, the value passed into a `DomainFunction` for that label
is a **tuple of 1D coordinate arrays** (one per spatial dimension), rather than a point cloud.

## Axis specs and quadrature metadata

Axis specs (`FourierAxisSpec`, `LegendreAxisSpec`, etc.) can attach an `AxisDiscretization` to the
batch, including:

- `nodes` (the axis coordinates),
- optional quadrature weights (for `integral`/`mean`),
- basis metadata used by `backend="basis"` differential operators.

This is how Phydrax keeps sampling, quadrature, and operator discretization consistent without
manual bookkeeping.

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
