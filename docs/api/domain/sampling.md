# Sampling

Sampling is an explicit plan-to-batch operation. Plans describe counts, layouts,
axis discretizations, and designs; batches carry sampled values plus named-axis
semantics for operators, models, constraints, and integration.

For the conceptual model, see
[Guides → Domains and sampling](../../guides_domain.md).

## Sampling plans

- `PointSampling` materializes a `PointBatch`. Its `SampleLayout` partitions
  non-fixed labels into jointly sampled blocks.
- `GridSampling` materializes a `GridBatch`. Its `axes` mapping requests
  coordinate axes for selected labels; its optional dense `PointSampling` handles
  remaining labels.

`TrajectoryDatasetDomain` and other coupled factors validate layouts
factor-wise. Their labels cannot be silently split across unrelated axes.

### Joint point designs

One layout block is one joint reference design. For
`SampleLayout((("x", "t"),))`, Phydrax generates one design spanning the
combined reference dimensions of `"x"` and `"t"` and maps its column slices
through target-measure transports. Fixed labels consume no reference dimensions.

```python
import jax.random as jr
import phydrax as phx

x = phx.domain.ScalarInterval(0.0, 1.0, label="x")
t = phx.domain.TimeInterval(0.0, 2.0)
component = (x @ t).component()

sampling = phx.domain.PointSampling(
    256,
    layout=phx.domain.SampleLayout((("x", "t"),)),
    design=phx.sampling.SobolDesign(scrambled=True),
)
batch = component.sample(sampling, key=jr.key(0))
```

IID and Latin-hypercube designs may compose independent native factor samplers
when no exact joint transport exists. Sobol, Halton, and Hammersley blocks reject
that case rather than silently replacing a multidimensional design with repeated
one-dimensional sequences.

### Axis-based grids

`GridSampling.axes` maps each gridded label to integer counts, axis specs, tuples
of axis specs, or a `GridSpec`. A dense point plan may cover labels that remain
paired:

```python
geom = phx.domain.Interval1d(0.0, 1.0)
batch = geom.component().sample(
    phx.domain.GridSampling(
        {"x": phx.domain.FourierAxisSpec(64)},
    ),
    key=jr.key(0),
)
```

`GridBatch` stores:

- `coord_axes_by_label`: named axes for each coordinate component;
- `coord_mask_by_label`: support masks on implied Cartesian grids;
- `coord_geometry_weight_by_label`: optional cut-cell corrections;
- `axis_discretization_by_axis`: nodes, quadrature weights, basis metadata, and
  nested-axis state;
- `dense_structure`: the canonical layout of any remaining point block.

Sampling describes sites and measures. Interpolation separately declares how
stored values are reconstructed at new sites.

## Core sampling API

::: phydrax.domain.SampleLayout
    options:
        members:
            - __init__
            - canonicalize
            - axis_for

---

::: phydrax.domain.PointSampling
    options:
        members:
            - __init__

---

::: phydrax.domain.GridSampling
    options:
        members:
            - __init__

---

::: phydrax.domain.PointBatch
    options:
        members:
            - __init__

---

::: phydrax.domain.GridBatch
    options:
        members:
            - __init__

## Typed reference designs

::: phydrax.sampling.IIDDesign

::: phydrax.sampling.LatinHypercubeDesign

::: phydrax.sampling.HammersleyDesign

::: phydrax.sampling.HaltonDesign

::: phydrax.sampling.SobolDesign

::: phydrax.sampling.DesignCapabilities

## Axis specs and grids

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
