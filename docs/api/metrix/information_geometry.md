# Hessian and Legendre information geometry

`HessianGeometry` generates local metric structure from a scalar potential `Φ` in one
explicit affine chart. `LegendreGeometry` strengthens that candidate with declared
primal and dual supports plus an explicit inverse gradient map. The distinction is
intentional: representative Hessian positivity does not prove global strict convexity,
essential smoothness, or invertibility.

For primal points `x` and `z`, the ordered Bregman divergence is

`DΦ(x ∥ z) = Φ(x) - Φ(z) - ⟨∇Φ(z), x - z⟩`.

It is generally asymmetric and does not satisfy a triangle inequality. Do not use it
as a geodesic distance. Locally,

`DΦ(x + δ ∥ x) = ½ ⟨δ, ∇²Φ(x) δ⟩ + O(||δ||³)`,

so the Hessian is the associated local Riemannian metric even though the finite
endpoint divergence carries additional affine-coordinate structure.

## Declaring Legendre duality

A Legendre geometry reuses `ChartSupport` and `ChartTransition`; it does not introduce
a second coordinate-domain convention. Supports preserve every leading batch axis and
classify the final chart axis pointwise.

```python
import jax.numpy as jnp
import phydrax as phx

primal_chart = phx.metrix.CoordinateChart(
    "positive",
    ("x0", "x1", "x2"),
)
dual_chart = phx.metrix.CoordinateChart(
    "log-positive",
    ("y0", "y1", "y2"),
)

hessian = phx.metrix.HessianGeometry(
    lambda x: jnp.sum(x * (jnp.log(x) - 1.0)),
    chart=primal_chart,
)
geometry = phx.metrix.LegendreGeometry(
    hessian,
    jnp.exp,
    primal_support=phx.metrix.ChartSupport(
        primal_chart,
        lambda x: jnp.all(x > 0.0, axis=-1),
        support_id="positive-orthant",
    ),
    dual_support=phx.metrix.ChartSupport(
        dual_chart,
        lambda y: jnp.all(jnp.isfinite(y), axis=-1),
        support_id="finite-log-coordinates",
    ),
    geometry_id="negative-entropy",
)
```

Here `∇Φ(x) = log(x)` and `(∇Φ)⁻¹(y) = exp(y)`. The resulting divergence is generalized
relative entropy:

`DΦ(x ∥ z) = Σᵢ [xᵢ log(xᵢ/zᵢ) - xᵢ + zᵢ]`.

`LegendreGeometry` never clips a zero coordinate, normalizes mass, jitters a matrix, or
numerically searches for a conjugate. Invalid primal and dual coordinates fail in both
eager and compiled execution.

## Dual potential and Fenchel--Young gap

For `y` in the declared dual support,

`Φ*(y) = ⟨(∇Φ)⁻¹(y), y⟩ - Φ((∇Φ)⁻¹(y))`.

`dual_potential(...)` evaluates this expression using the supplied inverse. The
Fenchel--Young gap

`Φ(x) + Φ*(y) - ⟨x, y⟩`

is zero when `y = ∇Φ(x)` and nonnegative for a genuine closed convex conjugate. The
dual Bregman orientation is

`DΦ*(u ∥ v) = DΦ((∇Φ)⁻¹(v) ∥ (∇Φ)⁻¹(u))`.

The implementation uses this endpoint identity rather than differentiating the dual
potential a second time.

## Direct dual translations

`dual_translate(x, s)` performs

`(∇Φ)⁻¹(∇Φ(x) + s)`.

It does not form a Hessian or solve a metric system. A mirror-descent update uses
`s = -αg`; see [Optimization](../optim.md#mirror-descent). The displacement must have
exactly the primal point shape and dtype. Leading axes remain independent products of
the final chart dimension.

## Local metrics and matrix-free actions

`metric()` returns the existing dense-coordinate `RiemannianMetric`. Use it only when
materializing a chart-sized metric is appropriate. `information_operator(...)` retains
the existing matrix-free Hessian action and bounded materialization contract.
Pass `precision=GeometryPrecisionPolicy(...)` to either the Hessian or Legendre
information-operator constructor to control coordinate validation and local compute
dtype. The resulting `InformationMetricOperator` retains the effective precision
evidence.

Damping changes the operator solve to a regularized preconditioner. It does not change
the Bregman divergence or declare a new potential.

## Validation

`validate_legendre_geometry(...)` evaluates representative points and reports:

- primal and dual support membership;
- Hessian symmetry, positive definiteness, and conditioning;
- primal and dual coordinate round trips;
- forward/inverse Jacobian consistency;
- Fenchel--Young equality on matched coordinates;
- diagonal Bregman residual;
- effective precision evidence.

A successful report is local evidence at the supplied points. It is not a proof of
global convexity, support convexity, or a global diffeomorphism.

## Numerical boundaries

- Negative entropy and logarithmic barriers have singular boundary gradients. Keep the
  domain open; use an explicit projection method when exact zeros are required.
- The direct Bregman formula subtracts nearly equal values near the diagonal. Small
  signed roundoff is reported rather than clamped.
- Bregman divergence depends on the declared affine structure. A nonlinear parameter
  pullback retains the local metric `JᵀGJ` but is not generally Bregman or dually flat
  in the parameter coordinates.
- KL, geodesic distance, Sinkhorn divergence, MMD, and score discrepancies keep their
  domain-specific APIs. Returning a scalar is not enough to give them one common
  semantic base.

## API

::: phydrax.metrix.HessianGeometry

---

::: phydrax.metrix.validate_hessian_geometry

---

::: phydrax.metrix.LegendreGeometry

---

::: phydrax.metrix.LegendreValidationReport

---

::: phydrax.metrix.validate_legendre_geometry
