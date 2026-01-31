# Integrals and measures

Phydrax's integral operators (`integral`, `mean`, and friends) estimate integrals over a
`DomainComponent` with a domain-aware measure.

## Measure \(\mu\) induced by the domain and component {: data-toc-label="Measure μ induced by the domain and component"}

Given a component \(\Omega_{\text{comp}}\subseteq \Omega\) (interior, boundary, fixed slice, etc.),
Phydrax interprets `integral(f, ..., component=...)` as estimating

$$
\int_{\Omega_{\text{comp}}} f(z)\,d\mu(z).
$$

The measure depends on the factor type and the component selection:

- **Geometry factors**:
  - `Interior()`: \(\mu(\Omega_x)\) is the geometry volume/area/length (`factor.volume`).
  - `Boundary()`: \(\mu(\partial\Omega_x)\) is the boundary measure (`factor.boundary_measure_value`).
  - `Fixed(...)`: counting measure on the fixed slice.
- **Scalar interval factors** (e.g. time):
  - `Interior()`: \(\mu(\Omega_t)\) is the interval measure (`factor.measure`).
  - `Boundary()`: defaults to 2-point counting measure (endpoints).
  - `FixedStart` / `FixedEnd` / `Fixed(...)`: counting measure on the fixed time.
- **Dataset factors** (`DatasetDomain`):
  - `Interior()`: either \(\mu(\Omega_{\text{data}})=1\) (`measure="probability"`) or
    \(\mu(\Omega_{\text{data}})=N\) (`measure="count"`).

## Sampling structure: paired vs coord-separable

Phydrax supports two structured sampling modes that affect how integrals are reduced:

- `PointsBatch` (paired sampling): you choose a `ProductStructure` and sample points per block.
- `CoordSeparableBatch` (coord-separable sampling): you sample 1D coordinate axes for selected geometry labels
  and evaluate on a Cartesian grid, optionally with per-axis discretization metadata.

### Default weights

If you do not supply an explicit `QuadratureBatch`, then:

For `PointsBatch`, Phydrax uses a uniform Monte Carlo-style weight on each sampling axis:

$$
w_a = \frac{\mu_a}{n_a},
$$

where \(n_a\) is the number of points on axis \(a\) and \(\mu_a\) is the product measure of the labels in that block.

For `CoordSeparableBatch`, Phydrax multiplies per-axis weights:

- if `AxisDiscretization.quad_weights` is present (e.g. Gauss–Legendre), those weights are used;
- otherwise weights fall back to uniform weights based on the factor's axis-aligned bounding box.

## Filtering and weighting (`where`, `weight_all`)

Integrals can be restricted and reweighted via the component:

- `component.where` / `component.where_all` act as indicator functions (masking out samples).
- `component.weight_all` multiplies the integrand by a user-defined weight field.

Conceptually, Phydrax estimates

$$
\int_{\Omega_{\text{comp}}} \mathbf{1}_{\text{where}}(z)\,w(z)\,f(z)\,d\mu(z).
$$

## Choosing reduction axes (`over=...`)

`over` controls which axes to integrate over:

- `over=None`: integrate over all axes implied by the batch.
- `over="x"`: integrate over the axis associated with label `"x"` (requires `"x"` to be a singleton block in paired sampling).
- `over=("x","t")`: integrate over a specific paired block.

For coord-separable batches, `over="x"` integrates over the coord-separable axes for that label.

## Examples

!!! example
    Gauss–Legendre quadrature on an interval:

    ```python
    import phydrax as phx

    geom = phx.domain.Interval1d(-1.0, 2.0)

    @geom.Function("x")
    def u(x):
        return x[0] ** 2

    batch = geom.component().sample_coord_separable({"x": phx.domain.LegendreAxisSpec(24)})
    val = phx.operators.integral(u, batch, component=geom.component())
    ```
