# Constraints and objectives

This guide explains how Phydrax turns PDE residuals, data fits, and integral targets into scalar
objective terms that can be summed and optimized.

## What a constraint is

At a high level, a constraint is a scalar loss term $\ell(\theta)$ computed from a set of
domain-aware field functions (often parameterized by neural network parameters $\theta$).
Solvers typically minimize a weighted sum:

$$
L(\theta) = \sum_i \ell_i(\theta).
$$

In Phydrax, constraints are objects with a common `.loss(functions, key=..., ...)` interface.

## Sampled (continuous) constraints

Many constraints are defined by:

1) a **domain component** (interior, boundary, initial slice, etc.), and  
2) a **residual operator** producing a `DomainFunction` $r(z)$ from one or more fields.

The pointwise penalty is a squared Frobenius norm:

$$
\rho(z) = \|r(z)\|_F^2 = \sum_i r_i(z)^2.
$$

Phydrax supports two reduction modes:

`reduction="mean"` (measure-normalized):

$$
\ell = w\,\frac{1}{\mu(\Omega_{\text{comp}})}\int_{\Omega_{\text{comp}}}\rho(z)\,\mathrm{d}\mu(z)
$$

`reduction="integral"` (unnormalized):

$$
\ell = w\int_{\Omega_{\text{comp}}}\rho(z)\,\mathrm{d}\mu(z)
$$

Here $\mu$ is the component measure induced by the domain (volume/area/length for interiors,
surface measure for boundaries, counting measure for fixed slices, etc.), and `weight = w`
is a scalar multiplier.

### Sampling structure and `over=...`

Sampling is controlled by a `ProductStructure` (paired point sampling) and optionally by
`coord_separable` (grid sampling).

The `over` argument selects which axes to reduce over:

- `over=None`: reduce over all sampled axes implied by the batch,
- `over="x"`: reduce over the axis for label `"x"` (when it is a singleton block in paired sampling),
- `over=("x", "t")`: reduce over a paired block.

For coord-separable batches, `over="x"` reduces over the coord-separable axes for that label.

### Filtering: `where` and `where_all`

Continuous constraints can restrict the sampling region via:

- `where={label: predicate}` (per-label filtering),
- `where_all=predicate` (global filtering on the full point tuple).

Conceptually this applies an indicator/mask inside the integral/mean.

## A common pattern: interior PDE residual

`ContinuousPointwiseInteriorConstraint` is a convenience wrapper for pointwise residual losses.

```python
import phydrax as phx

geom = phx.domain.Interval1d(0.0, 1.0)

@geom.Function("x")
def u(x):
    return x[0] ** 2

structure = phx.domain.ProductStructure((("x",),))
constraint = phx.constraints.ContinuousPointwiseInteriorConstraint(
    "u",
    geom,
    operator=lambda f: phx.operators.laplacian(f, var="x"),
    num_points=128,
    structure=structure,
    reduction="mean",
)
```

## Discrete and pointset constraints

For sensor/anchor data (discrete samples), Phydrax provides constraints that do not sample from a
component, but instead evaluate on explicit point sets (and typically reduce by mean/integral in
an analogous way).

## Integral equality constraints

Integral constraints enforce targets of the form

$$
\int_{\Omega_{\text{comp}}} f(z)\,d\mu(z) = c,
$$

where the left-hand side is estimated via Monte Carlo or quadrature, depending on the batch.
See [Guides → Integrals and measures](guides_integrals.md) for the measure/weighting details.
