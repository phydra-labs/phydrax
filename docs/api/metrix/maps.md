# Differentiable maps

`DifferentiableMap` represents a map between explicit source and target charts. It
supports unequal dimensions, Jacobians, Hessians, composition, tangent pushforward,
and covector pullback. An inverse is optional; APIs that require one reject maps that
do not provide it.

```python
import jax.numpy as jnp
import phydrax as phx

polar = phx.metrix.CoordinateChart("polar", ("r", "theta"))
cartesian = phx.metrix.CoordinateChart("cartesian", ("x", "y"))

polar_to_cartesian = phx.metrix.DifferentiableMap(
    polar,
    cartesian,
    lambda q: jnp.array([
        q[0] * jnp.cos(q[1]),
        q[0] * jnp.sin(q[1]),
    ]),
)

point = jnp.array([2.0, 0.3])
tangent = polar_to_cartesian.pushforward(point, jnp.array([0.1, -0.2]))
```

A metric pullback is defined for any differentiable map. Pulling back an affine
connection additionally requires an equal-dimensional locally invertible map because
connection coefficients obey an inhomogeneous transformation law.

::: phydrax.metrix.DifferentiableMap

::: phydrax.metrix.pullback_metric

::: phydrax.metrix.pullback_semi_riemannian_metric

::: phydrax.metrix.pullback_lorentzian_metric
