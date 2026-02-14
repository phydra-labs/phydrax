# Discrete constraints

For `PointSetConstraint`, runtime operator knobs can be set once via
`eval_kwargs` and are merged into each `.loss(...)` call. This is useful for
fixed-cloud MFD pointsets, e.g.

```python
import jax.numpy as jnp
import jax.random as jr
import phydrax as phx

geom = phx.domain.Interval1d(0.0, 1.0)
component = geom.component()
coords = jnp.linspace(0.0, 1.0, 32)
points = {"x": coords.reshape((-1, 1))}
plan = phx.operators.build_mfd_cloud_plan(points["x"], order=1, k=5)


def _x_values(x):
    if isinstance(x, tuple):
        return jnp.asarray(x[0]).reshape((-1,))
    arr = jnp.asarray(x)
    if arr.ndim == 1:
        return arr.reshape((-1,))
    return arr[:, 0]


@geom.Function("x")
def u(x):
    xx = _x_values(x)
    return xx**3


@geom.Function("x")
def dudx_exact(x):
    xx = _x_values(x)
    return 3.0 * xx**2


constraint = phx.constraints.PointSetConstraint.from_points(
    component=component,
    points=points,
    residual=lambda fns: phx.operators.partial_n(
        fns["u"], var="x", order=1, backend="mfd"
    )
    - dudx_exact,
    constraint_vars=("u",),
    eval_kwargs={"mfd_mode": "cloud", "mfd_cloud_plan": plan},
)

loss = constraint.loss({"u": u}, key=jr.key(0))
assert jnp.isfinite(loss)
```

For multi-axis cloud derivatives (for example Laplacians built from multiple
`partial_n` operators), pass a `(axis, order) -> plan` mapping via
`eval_kwargs={"mfd_mode": "cloud", "mfd_cloud_plans": plans}`.

## Discrete point constraints

::: phydrax.constraints.PointSetConstraint
    options:
        members:
            - __init__
            - from_points
            - from_operator
            - loss

---

::: phydrax.constraints.DiscreteInteriorDataConstraint

---

::: phydrax.constraints.DiscreteTimeDataConstraint

## Discrete boundary / initial constraints

::: phydrax.constraints.DiscreteDirichletBoundaryConstraint

---

::: phydrax.constraints.DiscreteNeumannBoundaryConstraint

---

::: phydrax.constraints.DiscreteInitialConstraint

## Discrete ODE constraints

::: phydrax.constraints.DiscreteODEConstraint
