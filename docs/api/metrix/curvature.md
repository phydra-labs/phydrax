# Curvature

Metrix computes curvature from the Levi-Civita connection with the convention

`R[..., l, k, i, j] = Rˡ_kij`.

The last two indices are the antisymmetric derivative pair. Ricci and scalar
curvature use direct contractions, so they do not materialize a full rank-four
Riemann tensor when only a contraction is requested.

::: phydrax.metrix.riemann_tensor

---

::: phydrax.metrix.ricci_tensor

---

::: phydrax.metrix.scalar_curvature

---

::: phydrax.metrix.einstein_tensor

---

::: phydrax.metrix.sectional_curvature

## Reference geometries

```python
import jax.numpy as jnp
import phydrax as phx

sphere = phx.metrix.CoordinateChart("sphere", ("theta", "phi"))
radius = 2.0
metric = phx.metrix.diagonal_metric(
    lambda q: radius**2 * jnp.array([1.0, jnp.sin(q[0]) ** 2]),
    chart=sphere,
)

q = jnp.array([1.1, 0.3])
scalar = phx.metrix.scalar_curvature(metric, q)  # 2 / radius**2
```

Useful validation identities include:

- Euclidean Cartesian and polar metrics have zero curvature.
- A radius-`R` two-sphere has `Ric = g/R²`, scalar curvature `2/R²`, and
  sectional curvature `1/R²`.
- The Poincaré half-plane metric `(dx² + dy²)/y²` has scalar curvature `-2`.
- The Riemann tensor is antisymmetric in its final two indices and satisfies the
  first Bianchi identity.

Full curvature is substantially more expensive than one contracted scalar.
Request `ricci_tensor`, `scalar_curvature`, or `sectional_curvature` directly
when that is the actual observable.
