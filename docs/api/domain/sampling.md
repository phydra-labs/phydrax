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
  geometry labels and evaluates on the implied Cartesian grid (with an interior mask).
  This is the natural mode for FFT/basis/spectral operators and neural operators (FNO, DeepONet).

Coord-separable sampling is driven by `DomainComponent.sample_coord_separable(...)`, which takes:

- `coord_separable`: a mapping from geometry label (e.g. `"x"`) to either
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

---

::: phydrax.domain.PointsBatch
    options:
        members:
            - __init__

---

::: phydrax.domain.QuadratureBatch
    options:
        members:
            - __init__
            - total_weight

---

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

---

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
