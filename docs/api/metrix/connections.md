# Connections and intrinsic operators

`LeviCivitaConnection` derives the unique torsion-free, metric-compatible
connection from a `RiemannianMetric`. Its coefficient convention is
`coefficients[..., k, i, j] = Γᵏᵢⱼ`.

::: phydrax.metrix.LeviCivitaConnection
    options:
        members:
            - __init__
            - coefficients
            - derivative

## Array-callable operators

The low-level Metrix operators accept a pointwise callable, a metric, and one
coordinate point or a leading batch of points.

::: phydrax.metrix.gradient

---

::: phydrax.metrix.covariant_hessian

---

::: phydrax.metrix.divergence

---

::: phydrax.metrix.covariant_derivative

---

::: phydrax.metrix.laplace_beltrami

For a tensor with component shape `(n,) × ... × (n,)`,
`covariant_derivative` appends the derivative axis. It supports scalar,
vector, covector, and mixed-tensor fields through an explicit `TensorType`.
Unsupported shapes fail rather than guessing which axes are geometric.

## Geodesics and parallel transport

The provided right-hand sides are solver-neutral JAX functions. Pass them to a
Diffrax ODE term or another integrator; Metrix does not own time stepping.

::: phydrax.metrix.geodesic_acceleration

---

::: phydrax.metrix.geodesic_rhs

---

::: phydrax.metrix.parallel_transport_rhs

## `DomainFunction` adapters

The corresponding Phydrax operators preserve domain labels, dense point batches,
coordinate-separable batches, and trainable metric leaves:

```python
import jax.numpy as jnp
import phydrax as phx

domain = phx.domain.Square(center=(2.0, 0.0), side=1.0)
chart = phx.metrix.CoordinateChart("polar", ("r", "theta"))
metric = phx.metrix.diagonal_metric(
    lambda q: jnp.array([1.0, q[0] ** 2]),
    chart=chart,
)
u = domain.Function("x")(lambda x: x[0] ** 2)
v = domain.Function("x")(lambda x: jnp.array([x[0], 0.0]))

riemannian_gradient = phx.operators.riemannian_grad(u, metric, var="x")
riemannian_divergence = phx.operators.riemannian_div(v, metric, var="x")
hessian = phx.operators.covariant_hessian(u, metric, var="x")
laplacian = phx.operators.laplace_beltrami(u, metric, var="x")
```

`phydrax.operators.laplace_beltrami` dispatches on its second argument:

- a `DomainComponent` selects the existing normal-based surface operator,
- a `RiemannianMetric` selects intrinsic coordinate calculus.

This keeps one mathematical operator name without conflating sampled extrinsic
surface normals with an intrinsic metric chart.

::: phydrax.operators.riemannian_grad

---

::: phydrax.operators.riemannian_div

---

::: phydrax.operators.riemannian_div_tensor

---

::: phydrax.operators.covariant_hessian

---

::: phydrax.operators.covariant_derivative

---

::: phydrax.operators.laplace_beltrami
