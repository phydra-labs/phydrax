# Domains and sampling

This guide explains Phydrax's *labeled domains* and the two explicit sampling
plans used throughout the library: paired `PointSampling` and axis-based
`GridSampling`.

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

## Geometry sources and domain adapters

Geometry construction is representation-aware and lives under `phx.geometry`.
Analytic, simplicial, B-Rep, CSG, and reconstructed sources all compile to the
same JAX-safe `CompiledGeometry` contract. `GeometryDomain` then adds the labeled
domain algebra used by fields, components, sampling, and integration:

```python
source = phx.geometry.Square(center=(0.0, 0.0), side=2.0)
geometry = source.compile()
space = phx.domain.GeometryDomain(geometry, label="x")
```

Apply geometry Boolean or transform operations to sources before compilation.
Apply `@`, `relabel`, `restrict`, and `component` to domain objects. Keeping these
roles separate prevents host-side CAD topology from leaking into JAX execution.


## Migration from the pre-refactor API

The geometry and domain cutover is intentionally direct; legacy aliases are not
retained. The canonical replacements are:

| Previous API | Canonical API |
| --- | --- |
| `phx.domain.Square(...)` and other 2D/3D constructors | `phx.domain.GeometryDomain(phx.geometry.Square(...).compile())` |
| `Geometry2DFromCAD(...)` / `Geometry3DFromCAD(...)` | `planar_region_from_source(...)`, `mesh_region_from_source(...)`, or `BRep(...)`, then `GeometryDomain(source.compile())` |
| Point-cloud, DEM, or LiDAR geometry constructors | `phx.geometry.reconstruct_planar_region(...)`, `reconstruct_surface_region(...)`, `reconstruct_dem_region(...)`, or `reconstruct_lidar_region(...)` |
| `ProductStructure(...)` | `SampleLayout(...)` |
| `component.sample(n, structure=layout, sampler=design)` | `component.sample(PointSampling(n, layout=layout, design=design))` |
| `component.sample_coord_separable(...)` | `component.sample(GridSampling({...}))` |
| `PointsBatch` / `CoordSeparableBatch` | `PointBatch` / `GridBatch` |
| `operator_domain_view_from_coord_separable(...)` | `operator_domain_view_from_grid(...)` |
| Inline `num_points=...`, `structure=...` sampling | `source=per_step(mean_over(component), PointSampling(..., layout=...))` |
| `Domain.Model(..., structured=True)` | `Domain.Model(..., binding=phx.nn.ModelBinding.axis())` |

`PointSampling` owns paired-site count and design. `GridSampling` owns named
axis specifications and an optional point plan for remaining labels. A
`SampleLayout` describes batch axes only; it never chooses points.


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
component = features.component()
batch = component.points(points)

@features.Function("x")
def observed(x):
    return jnp.sum(x)

observation = phx.conditions.Observation("u", component, observed)
source = phx.integration.fixed(
    phx.integration.from_samples(phx.integration.mean_over(component), batch)
)
data = phx.terms.ObservationPenalty(observation, source)
```

`points` is the raw `(N, d)` array. Do not wrap it in a dictionary unless the
domain is a product with multiple labels. The function receives each row as one
`(d,)` vector named `"x"`.

## Empirical dataset domains

Use `DatasetDomain` when the row itself is an empirical case or condition. This is
the right shape when rows are PyTrees, multimodal inputs, branch conditions, or
finite cases that will later be paired with physical coordinates.

For row-aligned scalar/vector targets on the empirical rows themselves, use
`SupervisedDatasetTerm`:

```python
import jax.numpy as jnp
import phydrax as phx

rows = jnp.asarray([[0.0, 1.0], [1.0, 2.0], [2.0, 4.0]])
targets = rows[:, 0] + 2.0 * rows[:, 1]
dataset_domain = phx.domain.DatasetDomain(rows)

@dataset_domain.Function("data")
def u(row):
    return row[0] + 2.0 * row[1]

term = phx.terms.SupervisedDatasetTerm(
    "u",
    dataset_domain.component(),
    targets,
    sampling=phx.domain.PointSampling(32, design="uniform"),
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

Because the row and time are coupled, use `SampleLayout((("data", "t"),))`
for trajectory point sampling, including fixed-time components.
The same paired batches support hard branch-conditional data enforcement through
`phx.enforcement.enforce_ragged_time_series`; choose `cubic_hermite`
interpolation when second time derivatives are part of the physics residual.

Keep static case features and time-varying signals separate. Store static scalars
or vectors in the `inputs` rows, and expose observed ragged signals with
`phx.terms.TrajectorySignal` when residuals need them. Per-case scalar/vector
labels should use `phx.terms.TrajectoryCaseDataTerm` rather than being repeated
as constant time series.

```python
import jax.numpy as jnp
import jax.random as jr
import phydrax as phx

inputs = jnp.asarray([[0.0], [1.0], [2.0]])
lengths = jnp.asarray([2, 4, 3])
trajectory_domain = phx.domain.TrajectoryDatasetDomain(inputs, lengths, dt=0.5)

component = trajectory_domain.component()
layout = phx.domain.SampleLayout((("data", "t"),))
sampling = phx.domain.PointSampling(8, layout=layout)
batch = component.sample(sampling, key=jr.key(0))

@trajectory_domain.Function("data", "t")
def u(data, t):
    return data[0] + t

values = u(batch)

condition = phx.conditions.Residual(
    "u",
    component,
    lambda u: phx.operators.partial_t(u, var="t") - 1.0,
)
source = phx.integration.per_step(
    phx.integration.mean_over(component),
    sampling,
)
physics = phx.terms.ResidualPenalty(condition, source)
```

```python
signal_values = (
    inputs[:, 0, None]
    + trajectory_domain.dt * jnp.arange(trajectory_domain.max_length)[None, :]
)
forcing = phx.terms.TrajectorySignal(
    trajectory_domain,
    signal_values,
    interpolation="linear",
)

targets = jnp.asarray([[1.0, 0.0], [2.0, -1.0], [3.0, -2.0]])
case_target = phx.terms.TrajectoryCaseDataTerm(
    "theta",
    trajectory_domain.component(),
    targets,
    sampling=phx.domain.PointSampling(32, design="uniform"),
)
```

Functions on a domain are wrapped as `DomainFunction`s. The key idea is that a `DomainFunction`
declares which labels it depends on, and operators, conditions, and terms use those labels consistently.

```python
@domain.Function("x", "t")
def u(x, t):
    return x[0] * (1.0 + t)
```
`Domain.Function(...)` wraps ordinary callables in a `PointwiseEvaluator`; a
callable may receive the keyword-only randomness key declared by its
`FunctionBinding`. Grid-native or graph-native execution uses an explicit
`BatchEvaluator`. `Domain.Model(...)` likewise requires the model's declared
`ModelBinding` (or an explicit binding for a plain callable). Evaluation never
switches protocol because of hidden call-time flags.



## Components: interior, boundary, and fixed slices

Conditions, penalties, and integrals are typically evaluated over a **domain
component**, which selects a subset of each factor:

- `Interior()`: the interior of a geometry or scalar interval;
- `Boundary()`: the boundary of a geometry or scalar interval (endpoints in 1D);
- `FixedStart()` / `FixedEnd()`: the start/end slice of a scalar interval (often time);
- `Fixed(value)`: a slice at a specified coordinate.

Components are created with `domain.component(...)`:

```python
# Continuing from: domain = geom @ time
component = domain.component({"t": phx.domain.FixedStart()})  # initial-time slice
```

A product domain's `boundary()` method returns a `ComponentSum`: one additive
term for each codimension-one face. This collection models a measure-disjoint
decomposition, so all terms share the same compatible labeled domain and exact
duplicates are rejected. It is not a geometric Boolean union; overlapping
filtered terms contribute once per term.

### Filtering with `where` and `where_all`

Sampling can be restricted by predicates:

- `where={label: predicate}` applies a per-label predicate, e.g. `where={"x": lambda x: x[0] < 0.5}`.
- `where_all=predicate` applies a predicate to the *full point tuple* (useful for coupled filters).

These filters behave like indicator functions: points that fail the predicate are
excluded from point samples or represented by a mask in grid samples.

## Paired point sampling (`PointBatch`)

Most pointwise PDE residual penalties use a `PointSampling` plan with a
`SampleLayout`.

A `SampleLayout` partitions sampled labels into jointly sampled blocks. Each block
corresponds to one named axis in the resulting `PointBatch`.

Examples:

- `SampleLayout((("x", "t"),))` samples paired space-time points.
- `SampleLayout((("x",), ("t",)))` samples space and time independently as a Cartesian product.

Within one block, Phydrax materializes one reference-space design whose dimension is
the sum of the active factors' reference dimensions, then slices its columns through
the factors' exact target-measure transports. A two-dimensional box paired with time
therefore consumes one three-dimensional design. Fixed labels consume no reference
dimensions. This distinction is essential for Sobol, Halton, and Hammersley designs:
several repeated one-dimensional sequences are not a multidimensional low-discrepancy
design.

IID sampling can fall back to independent native factor samplers because that still
produces the correct product measure. A non-IID multi-label block without exact
reference transports is rejected instead of silently weakening its design. Split such
labels into separate blocks when separate designs are intended.

```python
import jax.random as jr
import phydrax as phx

# Continuing from: domain = geom @ time
layout = phx.domain.SampleLayout((("x", "t"),))
batch = domain.component().sample(
    phx.domain.PointSampling(128, layout=layout),
    key=jr.key(0),
)
```

### Deterministic and scrambled low-discrepancy samplers

`design="halton"` and `design="sobol"` select the standard unscrambled
low-discrepancy sequences. Their points are independent of the supplied random
key or host seed. Use `"halton_scrambled"` or `"sobol_scrambled"` when randomized
scrambling is required; a fixed key or seed reproduces the same scrambled
sequence, while different keys or seeds produce different points. Host and JAX
sampling backends follow the same naming and reproducibility contract.

The equivalent typed forms live under `phx.sampling`:

```python
batch = domain.component().sample(
    phx.domain.PointSampling(
        128,
        layout=layout,
        design=phx.sampling.SobolDesign(scrambled=True),
    ),
    key=jr.key(0),
)
```

`phx.sampling.design_capabilities(...)` reports whether a design is randomized,
count-dependent, prefix-stable, random-access, factorwise-composable, or JAX-native.
IID and Latin-hypercube designs are factorwise-composable. Hammersley is finite and
count-dependent; Sobol and Halton are sequences whose joint dimensions must stay
together.

Per-label `where` and global `where_all` component predicates remain indicator masks
on the target measure. They do not condition the reference design by rejection.

## Axis-based grid sampling (`GridBatch`)

For spectral/basis operators and neural operators, sample one-dimensional axes
and evaluate on their implied Cartesian grid with `GridSampling`.

Pass a per-label axis request, such as
`{"x": FourierAxisSpec(64)}` for a one-dimensional periodic grid or
`{"x": (64, 64)}` for a two-dimensional grid:

```python
import jax.random as jr
import phydrax as phx

geom = phx.domain.Interval1d(0.0, 1.0)
batch = geom.component().sample(
    phx.domain.GridSampling({"x": phx.domain.FourierAxisSpec(64)}),
    key=jr.key(0),
)
```

`GridBatch` retains the coordinate axes, support masks, discretizations, and any
dense residual point block. Domain functions receive the canonical broadcast
implied by those axes; models that need a different contract must declare an
explicit `BatchEvaluator` or model binding.

## Axis specs and quadrature metadata

Axis specs (`FourierAxisSpec`, `LegendreAxisSpec`, etc.) can attach an `AxisDiscretization` to the
batch, including:

- `nodes` (the axis coordinates),
- optional quadrature weights (for `integral`/`mean`),
- basis metadata used by `backend="basis"` differential operators.

This is how Phydrax keeps sampling, quadrature, and operator discretization consistent without
manual bookkeeping.

## Sampling metadata and reconstruction

Grid sampling describes where values live. Deterministic
interpolation describes how stored values are reconstructed elsewhere. Phydrax
keeps these responsibilities separate: `AxisDiscretization` carries canonical
nodes and basis metadata, while trajectory, rectilinear, inverse-distance, and
Fourier adapters choose method-specific reconstruction rules.

For example, a rectilinear warp can normalize the physical nodes from an
`AxisDiscretization`, then apply its declared clamp, reflect, periodic, or
constant boundary mode. A Fourier transfer consumes a periodic sampled grid
but does not reinterpret its quadrature weights. Interpolated support is a
boolean statement about reconstruction; it is not a physical measure or an
integration weight.

See [API → Operators → Interpolation](api/operators/interpolation.md) for the
method taxonomy and numerical invariants.

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
cad = phx.domain.GeometryDomain(
    phx.geometry.Circle(center=(0.0, 0.0), radius=1.0).compile()
)
grid = phx.domain.GridSpec(
    (
        phx.domain.UniformAxisSpec(25),
        phx.domain.UniformAxisSpec(25),
    ),
    cut_cell_order=3,
)
batch = cad.component().sample(phx.domain.GridSampling({"x": grid}))
```

Each tensor node represents its bounding Voronoi subcell. Phydrax probes each
subcell, estimates its occupied fraction, keeps evaluation points inside the
declared geometry, and normalizes the represented mass to the geometry measure.
This improves constant/low-order integration over a binary nodal mask. It does
not recover a disconnected feature that has no interior tensor node. Use a
paired-point representation or a geometry-conforming mesh for such features.
`GridBatch` rejects negative, non-finite, misaligned, and unknown-label geometry
weights.

### Geometry boundary atlases and measure partitions

Every analytic, simplicial, or B-Rep geometry with boundary-atlas capability
exposes the same `boundary_atlas`. One-dimensional reference charts describe
planar boundary curves; two-dimensional charts describe solid surfaces. Each
chart carries its physical Jacobian, outward frame, optional trim domain, stable
source entity identity, physical tags, and seam ownership.

`phx.geometry.BoundaryAtlasPartition(cad.boundary_atlas)` computes a
physical-measure stratum for each chart and supports fixed-size stratified
sampling. `phx.geometry.GeometryMeasurePartition` is the lower-level explicit
partition for segment or triangle arrays. Interior volume cells are never
invented from a boundary-only surface mesh.

Fixed boundary integration lowers every chart through the unified integration
API. Reference nodes are mapped to physical edges or surface patches and
multiplied by the chart Jacobian and trim mask. Adjacent charts share only
measure-zero seams, so chart reduction does not double-count physical boundary
measure:

```py
import phydrax as phx

target = phx.integration.over(
    cad.component({"x": phx.domain.Boundary()})
)
plan = phx.integration.FixedQuadraturePlan(
    phx.integration.GaussLegendreRule(6)
)
surface_measure = phx.integration.integrate(1.0, target, plan).value
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

layout = phx.domain.SampleLayout((("x", "p"),))  # paired (x,p) samples
batch = phase.component().sample(phx.domain.PointSampling(256, layout=layout))
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

layout = phx.domain.SampleLayout((("x", "p", "t"),))
batch = phase_time.component().sample(
    phx.domain.PointSampling(512, layout=layout)
)
val = f(batch)
```

### Higher-dimensional momentum domains

For multi-dimensional momentum, relabel a 2D/3D geometry:

```python
import phydrax as phx

x = phx.domain.GeometryDomain(
    phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile()
)
p = phx.domain.GeometryDomain(
    phx.geometry.Square(center=(0.0, 0.0), side=6.0).compile(),
    label="p",
)
phase = x @ p
```
