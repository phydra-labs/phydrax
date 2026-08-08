# Interpolation

Interpolation reconstructs values stored at discrete source sites. It does not
choose those sites: domain sampling still owns point generation, joint design
semantics, masks, measures, and coord-separable axis metadata. Integration,
stochastic ancestry resampling, and conservative state transfer likewise retain
their own contracts.

## Native reconstruction substrate

Phydrax uses one private, JAX-native numerical substrate for deterministic
reconstruction while keeping the existing semantic APIs:

- trajectory and ragged-series adapters use explicit nearest, piecewise-linear,
  and local cubic-Hermite maps;
- point-data and latent-support adapters share normalized inverse-distance
  weights while retaining their geometry-specific candidate searches;
- rectilinear warps use multilinear maps with explicit periodic, reflected,
  clamped, or constant boundaries and explicit source-mask policy;
- FNO, CNO, and HOFNO resolution changes share one parity-correct Fourier
  transfer, including even-grid Nyquist splitting and merging;
- Smolyak interpolation retains its sparse combination structure and uses the
  shared barycentric basis primitive.
- rational CAD curves and surfaces use span-local B-spline maps with explicit
  derivative weights and exactly `degree + 1` active controls per parameter axis.

The common operation is a sparse or structured map from source values to query
values. Map construction owns indices, weights, validity, and support; map
application handles arbitrary trailing payload dimensions without replicating
source rows. Unsupported queries remain explicit. Mask behavior is selected as
strict rejection, valid-weight renormalization, or an adapter-owned fill value.

B-spline source values are control coefficients, not interpolation samples. An
expanded, nondecreasing knot vector and a degree define the active parameter
interval and the sparse control routes. Exact interior knots use the right-hand
span; the exact upper endpoint uses the final span and therefore has the
one-sided polynomial derivative rather than a constant endpoint branch. Value
weights form a partition of unity, while every positive-order derivative weight
sums to zero. Repeated knots and arbitrary trailing real or complex payload
dimensions remain supported.

`BSplineGrid(knots, degree)` is the canonical fixed-grid object shared by spline
edge models, projection, and fitting. It validates finite, nondecreasing knots
and legal multiplicities; exposes the coefficient count, active interval,
positive-span breakpoints, Greville abscissae, and interior continuity orders;
and generates Gauss--Legendre rules over the actual positive spans. Repeated
knots therefore do not create zero-width quadrature cells. Use
`BSplineGrid.open_uniform(degree, num_intervals, interval=(lower, upper))` only
when an open-uniform grid is intended.

The dynamic evaluator supplies an analytic piecewise-polynomial JVP with respect
to query coordinates. JVP, VJP, Jacobian, and higher query derivatives reuse the
same span-local basis jet; coefficient derivatives remain ordinary sparse
linear-map derivatives. Exact endpoints retain their selected one-sided
derivatives, while `clip` and `fill` queries strictly outside the active interval
have zero coordinate derivative.

This local map preserves the useful fixed-reconstruction-map idea from
[Splinex](https://github.com/cornelius-braun/splinex) without adding Splinex as a
dependency or adopting its dense basis matrices, implicit control padding, or
control-point-as-sample semantics.

Method-specific semantics are intentionally not hidden behind a universal
string-dispatched public function. Nearest ties, temporal extrapolation,
periodic seams, inverse-distance snap behavior, and conservative transfer are
different contracts. Use the domain, constraint, or neural-operator API that
owns those semantics.

Interpolation weights reproduce constants when they form a partition of unity.
That statement does not imply conservation against a physical measure.
Quadrature and conservative transfer remain separate measure-aware operations.

## B-spline fitting and smoothing

`fit_bspline(nodes, values, ...)` distinguishes observation samples from
B-spline control coefficients. It returns an immutable `BSplineInterpolant`
whose coefficient axis is leading and whose arbitrary trailing scalar, vector,
matrix, tensor, real, or complex payload shape is preserved.

The default plan performs exact cubic interpolation. Nodes may be unsorted;
Phydrax stably orders nodes and values together. Exact interpolation requires
distinct nodes and a square full-rank collocation system:

```python
import jax.numpy as jnp
import phydrax as phx

nodes = jnp.linspace(-1.0, 1.0, 9)
values = jnp.stack((nodes**3 - 2.0 * nodes, nodes**2), axis=-1)
curve = phx.operators.fit_bspline(nodes, values)

value = curve(jnp.asarray(0.2))
first = curve.derivative(jnp.asarray(0.2), 1)
```

Regression uses a deliberately smaller coefficient space:

```python
least_squares = phx.operators.fit_bspline(
    nodes,
    values,
    plan=phx.operators.BSplineInterpolationPlan(
        degree=3,
        num_intervals=4,
        mode="least_squares",
    ),
)
```

Nonnegative `sample_weights` scale observation residuals. Zero-weight
observations are retained in fit diagnostics but do not affect the solution.
Duplicate nodes are legal for least-squares and smoothing fits, but not for
exact interpolation. Underdetermined and rank-deficient systems fail with an
explicit error rather than silently selecting a minimum-norm interpolant.

`mode="smooth"` minimizes weighted data error plus an exact span-quadrature
Sobolev derivative energy. The smoothing value is the coefficient on that
physical-coordinate energy, so changing a grid's physical interval changes
derivative scaling correctly:

```python
smoothed = phx.operators.fit_bspline(
    nodes,
    values,
    plan=phx.operators.BSplineInterpolationPlan(
        degree=3,
        num_intervals=6,
        mode="smooth",
        smoothing=1e-3,
        regularization_order=2,
    ),
)
```

`boundary="open"` adds no fitting equations. `"natural"` imposes zero second
derivative at both endpoints. `"periodic"` equates endpoint derivatives from
order zero through `degree - 1`. Exact value or derivative jets use
`BSplineBoundaryConstraint("lower", order, value)`,
`BSplineBoundaryConstraint("upper", order, value)`, or a physical coordinate.
Constraints are solved as exact equations during regression and smoothing; they
are never represented by arbitrary penalty weights.

Every fit reports matrix rank, a condition estimate, weighted residual norm,
constraint residual norm, and unscaled regularization energy through
`interpolant.diagnostics`. An explicit `BSplineGrid` fixes the coefficient
space and physical active interval. Without one, the fitter constructs an
open-uniform grid over the minimum and maximum observation coordinates.

`interpolate_bspline(function, nodes, ...)` snapshots a one-dependency
`DomainFunction` through one named `PointBatch` and returns another
`DomainFunction` with the original domain, dependency, and metadata. The fitted
state is excluded from solver parameter partitions. Array evaluation and
automatic first or higher coordinate derivatives remain compatible with JAX
transforms.

## Fourier reconstruction

Periodic tensor grids have one coefficient substrate with three evaluation
paths:

- `fourier_resample(values, output_shape)` transfers a field to an aligned,
  endpoint-excluded uniform grid;
- `fourier_resample(..., phase_offsets=offsets)` evaluates a shifted uniform
  grid through coefficient-space phase factors and FFT resizing;
- `fourier_interpolate(values, coordinates, spatial_ndim=...)` evaluates paired
  arbitrary coordinates directly or with NUFFTAX Type 2.

These are reconstruction operations, not sampling policies. Domain sampling
still chooses sites, keys, masks, and measures. Fourier reconstruction consumes
stored periodic-grid values and query coordinates; it does not generate points
or infer source masks.

The low-level point evaluator uses
`values.shape = batch_shape + source_shape + payload_shape` and
`coordinates.shape = batch_shape + query_shape + (spatial_ndim,)`.
`payload_ndim` identifies the trailing payload axes. Batch shapes must match
exactly, so shared coordinates should be explicitly broadcast when values have
case axes. The public neural adapter specializes this to channel-last fields:

```python
import jax.numpy as jnp
import phydrax as phx

values = jnp.ones((8, 8, 1))
coordinates = jnp.asarray([[0.0, 0.0], [0.25, -0.5]])
sensor_coordinates = coordinates
x_nodes = jnp.linspace(-1.0, 1.0, 8, endpoint=False)
y_nodes = jnp.linspace(-1.0, 1.0, 8, endpoint=False)
x_period = 2.0
y_period = 2.0

sampled = phx.nn.layers.sample_fourier_grid(
    values,       # batch_shape + source_shape + (channels,)
    coordinates,  # batch_shape + query_shape + (spatial_ndim,)
    spatial_ndim=2,
)
```

Without explicit axis nodes, `sample_fourier_grid` uses the normalized periodic
axes `-1 + 2*j/n`, with period two. Physical axes are supplied as one strictly
increasing, endpoint-excluded uniform node vector and one positive period per
spatial dimension:

```python
sampled = phx.nn.layers.sample_fourier_grid(
    values,
    sensor_coordinates,
    spatial_ndim=2,
    axis_nodes=(x_nodes, y_nodes),
    periods=(x_period, y_period),
)
```

Queries are wrapped periodically. Consequently the support result is all true
after finite-input and axis validation:

```python
sampled, support = phx.nn.layers.sample_fourier_grid(
    values,
    sensor_coordinates,
    spatial_ndim=2,
    return_support=True,
)
```

`method="direct"` is the exact, roundoff-limited reference implementation and
is appropriate for small query sets or small mode products. It supports any
positive number of spatial dimensions. `method="nufft"` delegates one-, two-,
or three-dimensional point evaluation to NUFFTAX Type 2 and requires an
explicit approximation tolerance:

```python
sampled = phx.nn.layers.sample_fourier_grid(
    values,
    sensor_coordinates,
    spatial_ndim=2,
    method="nufft",
    tolerance=1e-6,
    query_chunk_size=4096,
)
```

`query_chunk_size` is static and uses padded `jax.lax.map` chunks, bounding the
largest point batch seen by either backend without exposing padded outputs.
NUFFT tolerance controls kernel construction; it is not a strict pointwise
error guarantee. Use the direct method as an oracle when calibrating a
tolerance for a new dtype, dimensionality, spectrum, or workload.

Even source axes need special treatment because one real-grid Nyquist
coefficient represents both signed frequencies away from the source grid.
Phydrax splits every even-axis Nyquist coefficient before shifted or arbitrary
evaluation, then merges target modes for regular-grid resampling. This preserves
odd/even, real/complex, and multidimensional Nyquist-corner semantics.

Fourier interpolation is global and signed. It therefore cannot be represented
faithfully as a local nonnegative `GatherStencil`, and it does not support
source-hole masking or local mask renormalization. Use rectilinear or
inverse-distance reconstruction for those contracts. NUFFT Type 1 is an adjoint
point-to-coefficient operation, not interpolation, and is intentionally not
part of this API.

## Smolyak interpolation

Phydrax fits reusable sparse polynomial surrogates as ordinary
`DomainFunction` objects. The fitted nodal state is immutable and excluded from
solver parameter partitions; evaluation remains compatible with JAX transforms
and Phydrax's named point batches.

## Fit and evaluate

```python
import jax.numpy as jnp
import phydrax as phx

x = phx.domain.ScalarInterval(-1.0, 1.0, label="x")
z = phx.domain.ProbabilityDomain(
    phx.uq.Normal(0.0, 1.0),
    label="z",
)
domain = x @ z

@domain.Function("x", "z")
def observable(x, z):
    return jnp.stack((x + z, x**2 + z**2))

plan = phx.operators.SmolyakInterpolationPlan(
    2,
    5,
    anisotropy=(1.0, 1.5),
    axis_rules="auto",
)
approximation = phx.operators.interpolate_smolyak(observable, plan)
value = approximation({"x": jnp.asarray(0.2), "z": jnp.asarray(-0.4)})
```

The plan's axis order is the source function's `deps` order, not necessarily the
containing domain's label order. Every dependency must be a scalar factor.
Factors that remain in the domain but are absent from `deps` are preserved and
are not sampled during fitting.

`axis_rules="auto"` selects:

- nested real Leja nodes for bounded scalar and uniform-reference probability
  factors;
- Gauss--Hermite nodes for standard-normal-reference probability factors.

Explicit choices are `"leja"`, `"clenshaw-curtis"`, and
`"gauss-hermite"`. Probability interpolation requires a producer-owned
bidirectional reference transform. Built-in `Uniform`, `Normal`, and
`LogNormal` distributions provide one; `EmpiricalDistribution` does not.

Fitting evaluates the source through one coupled `PointBatch`, coalescing all
structurally identical nested nodes first. The resulting
`SmolyakInterpolant` groups tensor terms by exact node-count signature and
vectorizes terms within each group. Scalar, vector, matrix, tensor, real, and
complex array outputs are retained without flattening.

The fit is eager and snapshots the source values. Repeated evaluations do not
retain or call the source function. Use `equinox.filter_jit` for complete
`DomainFunction`/`PointBatch` calls or ordinary `jax.jit` around an array-only
query wrapper. Automatic first and higher derivatives use the interpolating
polynomial, including at interpolation nodes.

## Integration

An interpolant has no separate integration convention. Integrate the returned
`DomainFunction` against an explicit Phydrax target and plan:

```python
estimate = phx.integration.integrate(
    approximation,
    phx.integration.over(domain.component()),
    phx.integration.SparseGridPlan(
        2,
        5,
        axis_rules=("clenshaw-curtis", "gauss-hermite"),
    ),
)
```

This keeps physical integrals, normalized expectations, density targets, and
integration diagnostics under the existing measure-aware integration
substrate.

The implementation independently adopts sparse-combination and grouped tensor
ideas described by [SmolyAX](https://github.com/JoWestermann/smolyax) and its
[JOSS paper](https://github.com/JoWestermann/smolyax/blob/main/paper/paper.md),
without adding SmolyAX as a dependency.

## API

::: phydrax.operators.BSplineInterpolationPlan

---

::: phydrax.operators.BSplineBoundaryConstraint

---

::: phydrax.operators.BSplineFitDiagnostics

---

::: phydrax.operators.BSplineInterpolant

---

::: phydrax.operators.fit_bspline

---

::: phydrax.operators.interpolate_bspline

---

::: phydrax.operators.SmolyakInterpolationPlan

---

::: phydrax.operators.SmolyakInterpolant

---

::: phydrax.operators.interpolate_smolyak
