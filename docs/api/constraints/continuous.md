# Continuous constraints

These helpers construct sampled constraints over domain components. For details on reduction,
measures, and filtering, see [Guides → Constraints and objectives](../../guides_constraints.md).

For residual-style continuous constraints, `weight` can be either a scalar global
multiplier or a pointwise `DomainFunction`.

## Interior / initial sampling constraints

::: phydrax.constraints.ContinuousPointwiseInteriorConstraint

---

::: phydrax.constraints.ContinuousInitialConstraint

---

::: phydrax.constraints.ContinuousInitialFunctionConstraint

## Integral constraints

Boundary-integral helpers accept `var` to select the geometry label. When it is
omitted, Phydrax infers the sole geometry label and raises on an ambiguous
domain. In a product domain, only that factor is changed to `Boundary()`; the
other factors retain their component defaults.

::: phydrax.constraints.ContinuousIntegralInteriorConstraint

---

::: phydrax.constraints.ContinuousIntegralBoundaryConstraint

---

::: phydrax.constraints.ContinuousIntegralInitialConstraint

## ODE constraints

::: phydrax.constraints.ContinuousODEConstraint

---

::: phydrax.constraints.InitialODEConstraint

## Stochastic evolution constraints

`ContinuousKolmogorovConstraint` builds either the stationary backward residual

\[
\mathcal L u=0
\]

when `evolution_var=None`, or

\[
\partial_tu+\mathcal L u=0
\]

when `evolution_var` names the time coordinate.
`ContinuousFokkerPlanckConstraint` similarly builds
\(\mathcal L^\ast p=0\) in stationary mode or

\[
\partial_tp-\mathcal L^\ast p=0
\]

in evolution mode. Both use `FunctionalConstraint`, so fixed/resampled batches,
sampling structures, weights, reductions, and adaptive collocation have the
same meaning as for other continuous residuals.

Passing `metric=` selects the covariant generator and Riemannian-volume
Fokker--Planck adjoint. Drift inputs retain coordinate Itô semantics and are
converted internally; with `interpretation="stratonovich"`, the coordinate
Stratonovich correction is applied first. See
[API → Metrix → Stochastic geometry](../metrix/stochastic.md).

Drift, diffusion, and covariance coefficients can be supplied as
`DomainFunction` objects or as names in the solver's `functions` mapping. A
named coefficient is part of `constraint_vars`, so its trainable leaves are
optimized jointly with the primary field:

```python
import jax.numpy as jnp
import phydrax as phx

state = phx.domain.Interval1d(-1.0, 1.0)
time = phx.domain.TimeInterval(0.0, 1.0)
domain = state @ time
u = domain.Function("x", "t")(
    lambda x, t: jnp.exp(-(x[0] ** 2) - t)
)
drift = domain.Function("x", "t")(
    lambda x, t: jnp.asarray([-x[0]])
)

scale = domain.Parameter(0.2)
sigma = scale * jnp.ones((1, 1))

constraint = phx.constraints.ContinuousKolmogorovConstraint(
    "u",
    domain.component(),
    drift=drift,
    diffusion="sigma",
    evolution_var="t",
    num_points=64,
    structure=phx.domain.ProductStructure((("x", "t"),)),
)
solver = phx.solver.FunctionalSolver(
    functions={"u": u, "sigma": sigma},
    constraints=[constraint],
)
```

These constraints enforce only the stochastic evolution equation. A probability
density's positivity, normalization, initial data, and absorbing, reflecting,
or zero-flux boundary behavior are separate contracts. Use a positive output
activation or ansatz, an integral equality constraint for normalization, and
the corresponding initial/boundary constraints. This separation prevents the
dynamics residual from silently changing the density model.

::: phydrax.constraints.ContinuousKolmogorovConstraint

---

::: phydrax.constraints.ContinuousFokkerPlanckConstraint
