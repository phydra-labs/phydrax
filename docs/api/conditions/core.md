# Core conditions and penalties

Conditions are declarative: they name fields, support, and scientific operators,
but do not sample, integrate, weight, or choose soft versus exact treatment.

## Generic conditions

::: phydrax.conditions.Residual
    options:
        members:
            - __init__
            - residual

---

::: phydrax.conditions.Moment
    options:
        members:
            - __init__
            - integrand

---

::: phydrax.conditions.Observation
    options:
        members:
            - __init__
            - residual

## Generic penalties

Residual and observation penalties integrate a squared pointwise mismatch.
Moment penalties integrate first and then square the mismatch with the
condition's target. Each receives an explicit integration source.

::: phydrax.terms.ResidualPenalty
    options:
        members:
            - __init__
            - pointwise_score
            - loss

---

::: phydrax.terms.MomentPenalty
    options:
        members:
            - __init__
            - loss

---

::: phydrax.terms.ObservationPenalty

## Supplied observations

`Observation` declares a target field independently of how its support is
realized. Bind explicit coordinates with `component.points(...)`, attach
target-measure weights with `integration.from_samples`, and choose fixed or
caller-owned realization lifetime:

```python
import jax.numpy as jnp
import phydrax as phx

space = phx.domain.Interval1d(0.0, 1.0)
component = space.component()
observed_x = jnp.linspace(0.0, 1.0, 8)[:, None]
target = space.Function("x")(lambda x: x[0] ** 2)
observation = phx.conditions.Observation("u", component, target)
batch = observation.on.points({"x": observed_x})
realization = phx.integration.from_samples(
    phx.integration.mean_over(observation.on),
    batch,
)
term = phx.terms.ObservationPenalty(
    observation,
    phx.integration.fixed(realization),
    scale=5.0,
)
```

Use `caller(phx.integration.mean_over(observation.on))` when an outer training
or evaluation loop owns the realization.

## Solver composition

```python
u = space.Function("x")(lambda x: x[0] ** 2)
solver = phx.solver.FunctionalSolver(
    functions={"u": u},
    terms=(term,),
    enforcement=None,
)
```

See [Integration sources](../integration.md#term-integration-sources) for
realization ownership and [Exact enforcement](../enforcement.md) for compiled transforms.
