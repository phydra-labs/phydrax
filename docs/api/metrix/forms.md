# Differential forms

`DifferentialForm` carries a canonical `phydrax.exterior.FormType`. Increasing
coordinate multi-indices use lexicographic order. Coefficients have shape
`(*batch, choose(n, k), *fiber_shape)`; the component axis remains present for
zero-forms and top forms.

The exterior derivative, wedge product, pullback and interior product are
metric-independent. Hodge star, codifferential and Hodge Laplacian require a
metric, not an orientation argument. Star flips untwisted/twisted parity.
Signed metrics retain signature-aware signs; no positive-definite norm is inferred.

For dimension `n`, form degree `k`, and metric index `q` (the number of negative
directions), the conventions are

```text
⋆⋆ α = (−1)^(k(n−k)+q) α
δ α = (−1)^(n(k+1)+q+1) ⋆ d ⋆ α
Δ_H α = d δ α + δ d α.
```

Thus `hodge_laplacian` on a zero-form is the negative Laplace--Beltrami operator
in positive-definite signature and the negative d'Alembertian in Lorentzian
signature. `to_untwisted(form, orientation)` and `to_twisted(form, orientation)`
perform explicit orientation-dependent conversions; δ itself is orientation-free.
The codifferential of a zero-form raises `ValueError`.

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
batch axes and trainable callable state. Both smooth carriers use the exterior
algebra kernel rather than independent sign tables. Continuous forms connect to
cochains through `phydrax.exterior.DeRhamBridge`, with explicit oriented cell maps
and quadrature. `integrate_form` returns a `DiscreteForm`, and
`validate_de_rham_commutation` measures the actual smooth/discrete commutator.
Refinement never silently changes quadrature.


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

::: phydrax.metrix.to_untwisted

::: phydrax.metrix.to_twisted

::: phydrax.exterior.DeRhamBridge

::: phydrax.exterior.integrate_form

::: phydrax.exterior.validate_de_rham_commutation
