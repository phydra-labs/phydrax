# Composition

Domains can be combined into product domains (e.g. space-time) and then used to define
domain-aware functions and constraints.

## Dataset factors (\(\Omega_{\text{data}}\)) {: data-toc-label="Dataset factors (Ω_data)"}

Many operator-learning workflows decompose the domain as a product

$$
\Omega = \Omega_{\text{data}} \times \Omega_x \times \Omega_t \times \cdots,
$$

where \(\Omega_{\text{data}}\) indexes a finite dataset of input functions/fields
(e.g. forcing terms, initial conditions, material parameters) and the remaining
factors are geometric/scalar coordinates.

`DatasetDomain` is an atomic joint factor that stores an in-memory PyTree whose
leaves share one leading dataset axis. It composes with physical factors through
`@` and participates in the same explicit sampling plans:

- `PointSampling(..., layout=SampleLayout(...))` for paired or independent
  point blocks;
- `GridSampling(..., dense=PointSampling(...))` when coordinate grids are paired
  with empirical rows;
- `Domain.Model(...)` for models with a declared `ModelBinding`.

Phydrax models expose their own input binding. A plain callable model must receive
an explicit `phx.nn.models.ModelBinding`; evaluation does not inspect signatures or
switch between flat, structured, pointwise, blockwise, or axis-batch execution
implicitly.

For row-indexed time series with different sequence lengths, use
[`TrajectoryDatasetDomain`](trajectory_dataset.md). It is not a rectangular
product: its dataset row and sampled time remain one inseparable joint factor.

### Measure semantics

`DatasetDomain(..., measure=...)` controls the measure used by integral/mean reductions:

- `measure="probability"`: \(\mu(\Omega_{\text{data}})=1\) (treat as an expectation),
- `measure="count"`: \(\mu(\Omega_{\text{data}})=N\) where \(N\) is dataset size
  (treat as a finite-sum domain).

!!! example
    ```python
    import jax.numpy as jnp
    import phydrax as phx

    data = jnp.ones((128, 64))  # N=128 samples, each with 64 features
    Omega = phx.domain.DatasetDomain(data, label="data") @ phx.domain.Interval1d(0.0, 1.0)
    ```

::: phydrax.domain.ProductDomain
    options:
        members:
            - __init__
            - factors
            - joint_factors
            - labels
            - factor
            - same_support
            - schema_compatible
            - restrict
            - drop
            - relabel
            - boundary

---

::: phydrax.domain.DatasetDomain
    options:
        members:
            - __init__
            - size
            - measure
            - sample
