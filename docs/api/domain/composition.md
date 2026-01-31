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

`DatasetDomain` is a unary domain that stores an in-memory dataset (a PyTree of arrays
with a shared leading dataset axis) and samples by random indexing. This integrates
cleanly with:

- `ProductDomain` composition via `@`,
- structured sampling (`dense_structure=ProductStructure((("data",),))`),
- `Domain.Model(...)` for building operator-learning models that take `(data, coords...)`.

### Measure semantics

`DatasetDomain(..., measure=...)` controls the measure used by integral/mean reductions:

- `measure="probability"`: \(\mu(\Omega_{\text{data}})=1\) (treat as an expectation),
- `measure="count"`: \(\mu(\Omega_{\text{data}})=N\) where \(N\) is dataset size
  (treat as a finite-sum domain).

!!! example
    ```python
    import jax.numpy as jnp
    import phydrax as phx

    data = jnp.ones((128, 64))     # N=128 samples, each with 64 features
    Omega = phx.domain.DatasetDomain(data, label="data") @ phx.domain.Interval1d(0.0, 1.0)
    ```

::: phydrax.domain.ProductDomain
    options:
        members:
            - __init__
            - factors
            - labels
            - factor
            - equivalent
            - boundary

---

::: phydrax.domain.DatasetDomain
    options:
        members:
            - __init__
            - size
            - measure
            - sample
