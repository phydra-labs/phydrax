# Sampling

Structured sampling yields batches that preserve axis meaning (via named axes), so that
operators and constraints can keep shape semantics without manual broadcasting.

For a conceptual overview (product structures, components, and coord-separable grids), see
[Guides → Domains and sampling](../../guides_domain.md).

## Paired vs coord-separable sampling

Phydrax supports two complementary structured sampling modes:

- **Paired sampling** (`PointsBatch`): samples *points* in each block of a `ProductStructure`.
  This is the default mode used by most pointwise PDE residual constraints.
- **Coord-separable sampling** (`CoordSeparableBatch`): samples *1D coordinate axes* for selected
  unary labels (geometry and/or scalar intervals) and evaluates on the implied Cartesian grid (with an interior mask).
  This is the natural mode for FFT/basis/spectral operators and neural operators (FNO, DeepONet).

`TrajectoryDatasetDomain` is paired-only: its dataset row and time label must stay
on the same sampling axis.

Sampling and interpolation are adjacent but distinct. Sampling returns source
sites, masks, axis identities, and measure metadata. Interpolation may later
reconstruct stored values at query sites, but it neither changes the sampled
measure nor supplies quadrature weights. A `CoordSeparableBatch` provides the
canonical source-axis metadata used by structured consumers; each consumer
still chooses its explicit boundary and support policy.

## Joint block designs

A paired block is also one joint reference design. For
`ProductStructure((("x", "t"),))`, Phydrax generates one design spanning the
combined reference dimensions of `"x"` and `"t"` and then maps its column slices
through exact target-measure transports. Fixed labels consume no dimensions.

Supported exact transports include scalar intervals, probability inverse CDFs,
`Interval1d`, `HyperRectangle` interiors and boundaries, and finite
`DatasetDomain` rows. IID and Latin-hypercube designs may use independent native
factor samplers when an exact transport is unavailable: both preserve their design
contract under factorwise composition. Sobol, Halton, and Hammersley multi-label
blocks reject that case because factorwise sequences would not preserve the
requested joint design.

Component `where` and `where_all` predicates remain target-measure masks; paired
sampling does not reinterpret them as rejection-conditioning predicates.

Typed designs and string shorthands are equivalent:

```python
import jax.random as jr
import phydrax as phx

x = phx.domain.ScalarInterval(0.0, 1.0, label="x")
t = phx.domain.TimeInterval(0.0, 2.0, label="t")
component = (x @ t).component()

typed = phx.sampling.SobolDesign(scrambled=True)
batch = component.sample(
    256,
    structure=phx.domain.ProductStructure((("x", "t"),)),
    sampler=typed,
    key=jr.key(0),
)
```

Use `phx.sampling.design_capabilities(typed)` to inspect randomized,
count-dependent, prefix-stable, random-access, factorwise-composable, and
JAX-native properties.

Coord-separable sampling is driven by `DomainComponent.sample_coord_separable(...)`, which takes:

- `coord_separable`: a mapping from unary label (e.g. `"x"` or `"t"`) to either
  counts (`int` / `Sequence[int]`) *or* basis-aware axis specs (`AbstractAxisSpec` implementations / `GridSpec`);
- `dense_structure` + `num_points`: how to sample any remaining non-fixed, non-separable labels
  (e.g. `"data"` for operator-learning datasets).

!!! example
    Coord-separable grid evaluation on an interval:

    ```python
    import jax.random as jr
    import phydrax as phx

    geom = phx.domain.Interval1d(0.0, 1.0)
    component = geom.component()

    batch = component.sample_coord_separable(
        {"x": phx.domain.FourierAxisSpec(64)},
        key=jr.key(0),
    )
    ```

!!! note
    A `CoordSeparableBatch` stores:

    - `coord_axes_by_label`: per-label axis names (for shape/dims inference),
    - `coord_mask_by_label`: per-label interior masks on the Cartesian grid,
    - `axis_discretization_by_axis`: optional per-axis metadata (nodes/weights/basis),
      used by quadrature and basis backends.

::: phydrax.domain.ProductStructure
    options:
        members:
            - __init__
            - canonicalize
            - axis_for

## Typed reference designs

::: phydrax.sampling.IIDDesign

::: phydrax.sampling.LatinHypercubeDesign

::: phydrax.sampling.HammersleyDesign

::: phydrax.sampling.HaltonDesign

::: phydrax.sampling.SobolDesign

::: phydrax.sampling.DesignCapabilities

---

::: phydrax.domain.PointsBatch
    options:
        members:
            - __init__


::: phydrax.domain.CoordSeparableBatch
    options:
        members:
            - __init__

---

## Coord-separable grids

::: phydrax.domain.AxisDiscretization
    options:
        members:
            - __init__

---

::: phydrax.domain.AbstractAxisSpec
    options:
        members:
            - __init__

---

::: phydrax.domain.GridSpec
    options:
        members:
            - __init__

---

::: phydrax.domain.UniformAxisSpec
    options:
        members:
            - __init__

---

::: phydrax.domain.FourierAxisSpec
    options:
        members:
            - __init__

---

::: phydrax.domain.SineAxisSpec
    options:
        members:
            - __init__

---

::: phydrax.domain.CosineAxisSpec
    options:
        members:
            - __init__

---

::: phydrax.domain.LegendreAxisSpec
    options:
        members:
            - __init__

::: phydrax.domain.NestedDyadicAxisSpec
    options:
        members:
            - __init__

---

## Geometry boundary structures

Every compiled geometry with boundary-atlas capability exposes
`phydrax.geometry.BoundaryAtlas`. Charts carry reference-to-physical maps,
Jacobians, outward frames, trim domains, source entity identities, physical tags,
and seam ownership. The same structure drives boundary sampling and fixed
quadrature for analytic, simplicial, and B-Rep geometry.

::: phydrax.geometry.BoundaryAtlas
    options:
        members:
            - __init__
            - num_charts
            - reference_dimension
            - map
            - jacobian
            - frame
            - select

---

::: phydrax.geometry.BoundaryAtlasPartition
    options:
        members:
            - __init__
            - sample

---

::: phydrax.geometry.GeometryMeasurePartition
    options:
        members:
            - __init__
            - sample

## Axis conventions (nodes + weights)

Many basis-aware operators (spectral derivatives, quadrature) want both:

- nodes \(x_j\) on an axis \([a,b]\),
- quadrature weights \(w_j\) to approximate \(\int_a^b f(x)\,dx \approx \sum_j w_j f(x_j)\).

When you sample with axis specs (`AbstractAxisSpec` implementations) / `GridSpec`, Phydrax materializes an `AxisDiscretization`
and attaches it on the batch (so downstream operators can reuse nodes/weights).

### Fourier (periodic, endpoint excluded)

For `FourierAxisSpec(n)`:

$$
x_j = a + (b-a)\frac{j}{n},\quad j=0,\dots,n-1,\qquad
w_j = \frac{b-a}{n}.
$$

### Sine (cell-centered interior grid)

For `SineAxisSpec(n)`:

$$
x_j = a + (b-a)\frac{j+\tfrac12}{n},\quad j=0,\dots,n-1,\qquad
w_j = \frac{b-a}{n}.
$$

### Cosine (endpoint grid + trapezoid weights)

For `CosineAxisSpec(n)`:

$$
x_j = a + (b-a)\frac{j}{n-1},\quad j=0,\dots,n-1,
$$

and trapezoid weights \(w_0=w_{n-1}=\tfrac12\Delta x\), \(w_j=\Delta x\) otherwise.

### Legendre (orthax Gauss / Radau / Lobatto)

For `LegendreAxisSpec(n)`, orthax produces nodes \(\xi_j\in[-1,1]\) and weights \(w_j\)
for the canonical interval. Phydrax maps them to \([a,b]\) via

$$
x_j=\tfrac{b-a}{2}\,\xi_j+\tfrac{a+b}{2},\qquad
\tilde w_j=\tfrac{b-a}{2}\,w_j.
$$
