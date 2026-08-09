# Charts and tensors

## Coordinate charts

`CoordinateChart` is a small static identity: a name plus an ordered tuple of
coordinate names. It deliberately has no bounds, seam policy, sampling scheme,
or unit registry.

::: phydrax.metrix.CoordinateChart
    options:
        members:
            - __init__
            - dimension
            - compatible_with

## Directed chart transitions

A `ChartTransition(source, target, map, inverse=...)` stores the directed map
`y = map(q)`. Its Jacobian is `J^a_i = ∂y^a/∂q^i`; its Hessian appends both
source-coordinate derivative axes. Composition is checked by chart identity, so
an unlabeled Jacobian with the wrong direction cannot be silently accepted.

```python
import jax.numpy as jnp
import phydrax as phx

polar = phx.metrix.CoordinateChart("polar", ("r", "theta"))
cartesian = phx.metrix.CoordinateChart("cartesian", ("x", "y"))

polar_to_cartesian = phx.metrix.ChartTransition(
    polar,
    cartesian,
    lambda q: jnp.array([q[0] * jnp.cos(q[1]), q[0] * jnp.sin(q[1])]),
    inverse=lambda x: jnp.array([jnp.linalg.norm(x), jnp.arctan2(x[1], x[0])]),
)
```

::: phydrax.metrix.ChartTransition
    options:
        members:
            - __init__
            - identity
            - __call__
            - inverse
            - jacobian
            - inverse_jacobian
            - hessian
            - compose

## Tensor types

`TensorType` declares the variance of every trailing component axis and an
optional density weight. It is metadata for transformation and contraction
kernels; it does not wrap the array or allocate a second tensor container.

```python
scalar = phx.metrix.TensorType()
vector = phx.metrix.TensorType(("contravariant",))
covector = phx.metrix.TensorType(("covariant",))
mixed = phx.metrix.TensorType(("contravariant", "covariant"))
density = phx.metrix.TensorType(density_weight=1.0)
```

For `y = y(q)`:

- contravariant axes transform with `J`,
- covariant axes transform with `J⁻ᵀ`,
- a density of weight `w` receives `|det J|⁻ʷ`.

The canonical constants are `SCALAR_TENSOR`, `VECTOR_TENSOR`,
`COVECTOR_TENSOR`, and `DENSITY_TENSOR`.

::: phydrax.metrix.TensorType

---

::: phydrax.metrix.reexpress_tensor

---

::: phydrax.metrix.pushforward_vector

---

::: phydrax.metrix.pullback_covector

## Index operations and contractions

Raising, lowering, and metric self-contraction accept every nondegenerate metric
signature. Despite its historical name, `tensor_norm_squared` is a self-contraction:
it can be negative or zero for a nonzero tensor under a signed metric and must not be
used as a positive norm in optimization or probabilistic code.

::: phydrax.metrix.raise_index

---

::: phydrax.metrix.lower_index

---

::: phydrax.metrix.contract_indices

---

::: phydrax.metrix.inner_product

---

::: phydrax.metrix.tensor_norm_squared
