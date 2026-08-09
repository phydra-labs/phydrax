# Differential forms

`DifferentialForm` stores only strictly increasing coordinate multi-indices. A degree
`k` form in dimension `n` therefore has `n choose k` coefficients rather than a dense
rank-`k` antisymmetric tensor. Batch axes remain leading axes.

The exterior derivative, wedge product, pullback, and interior product are
metric-independent. The Hodge star, codifferential, and Hodge--de Rham Laplacian
require an explicit metric and orientation. Signed metrics use signature-aware Hodge
semantics; no positive-definite norm is inferred.

For dimension `n`, form degree `k`, and metric index `q` (the number of negative
directions), the conventions are

```text
⋆⋆ α = (−1)^(k(n−k)+q) α
δ α = (−1)^(n(k+1)+q+1) ⋆ d ⋆ α
Δ_H α = d δ α + δ d α.
```

Thus `hodge_laplacian` on a zero-form is the negative Laplace--Beltrami operator
in positive-definite signature and the negative d'Alembertian in Lorentzian
signature. Orientation is explicit; using the same orientation in both Hodge
stars makes the codifferential orientation-independent.

```python
import jax.numpy as jnp
import phydrax as phx

chart = phx.metrix.CoordinateChart("plane", ("x", "y"))
alpha = phx.metrix.DifferentialForm(
    lambda q: jnp.array([-q[1], q[0]]),
    chart=chart,
    degree=1,
)

d_alpha = phx.metrix.exterior_derivative(alpha)
assert jnp.allclose(d_alpha(jnp.array([0.2, 0.3])), jnp.array([2.0]))
```

`DomainDifferentialForm` carries the same form semantics through labeled
`DomainFunction` programs. Its Hodge star, codifferential, and Hodge Laplacian accept
the same positive-definite or signed metric objects while preserving dependencies,
batch axes, and trainable callable state. Continuous forms connect to metric cochains
only through an explicit `ContinuousCochainBridge` containing oriented cell
parameterizations and quadrature.
Refinement does not silently change the declared quadrature. Projection
convergence is therefore a property of the supplied cell maps and rule, while
`validate_stokes_bridge` measures the actual smooth/discrete exterior-derivative
commutator. The reference uniform-segment coverage verifies second-order global
midpoint projection and third-order cellwise Stokes residual decay.


## Maxwell residual composition

On a four-dimensional Lorentzian chart, `domain_maxwell_residuals` composes a
degree-two field strength \(F\) into

\[
dF-M,\qquad \delta F+J^\flat.
\]

Here \(M\) is an optional magnetic-current three-form. \(J^\flat\) is the
physical electric-current covector; the plus sign follows Phydrax's declared
codifferential convention, for which
\((\delta F)_\nu=-\nabla^\mu F_{\mu\nu}\).
When \(F=dA\), the homogeneous vacuum residual vanishes by \(d^2=0\).
The returned degree-three and degree-one forms expose ordinary
`DomainFunction` coefficients, so each can be returned directly from a
`phydrax.conditions.Residual` operator and reduced by `ResidualPenalty`.

::: phydrax.operators.DomainMaxwellResiduals

::: phydrax.operators.domain_maxwell_residuals

::: phydrax.metrix.DifferentialForm

::: phydrax.metrix.exterior_derivative

::: phydrax.metrix.wedge

::: phydrax.metrix.pullback_form

::: phydrax.metrix.interior_product

::: phydrax.metrix.lie_derivative

::: phydrax.metrix.hodge_star

::: phydrax.metrix.codifferential

::: phydrax.metrix.hodge_laplacian

::: phydrax.operators.DomainDifferentialForm

::: phydrax.operators.domain_hodge_star

::: phydrax.operators.domain_codifferential

::: phydrax.operators.domain_hodge_laplacian

::: phydrax.graph.ContinuousCochainBridge

::: phydrax.graph.integrate_form_to_cochain

::: phydrax.graph.validate_stokes_bridge
