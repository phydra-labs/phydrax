# Differential forms

`DifferentialForm` stores only strictly increasing coordinate multi-indices. A degree
`k` form in dimension `n` therefore has `n choose k` coefficients rather than a dense
rank-`k` antisymmetric tensor. Batch axes remain leading axes.

The exterior derivative, wedge product, pullback, and interior product are
metric-independent. The Hodge star, codifferential, and Hodge--de Rham Laplacian
require an explicit metric and orientation. Signed metrics use signature-aware Hodge
semantics; no positive-definite norm is inferred.

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
`DomainFunction` programs. Continuous forms connect to metric cochains only through
an explicit `ContinuousCochainBridge` containing oriented cell parameterizations and
quadrature.

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

::: phydrax.graph.ContinuousCochainBridge

::: phydrax.graph.integrate_form_to_cochain

::: phydrax.graph.validate_stokes_bridge
