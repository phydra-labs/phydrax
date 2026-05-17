# Discrete constraints

For `PointSetConstraint`, runtime operator knobs can be set once via
`eval_kwargs` and are merged into each `.loss(...)` call.

`PointSetConstraint.weight` can be a scalar global multiplier or a
`DomainFunction` evaluated pointwise on the anchor set.

Data-fit constraints created by `DiscreteInteriorDataConstraint` and
`DiscreteTimeDataConstraint` attach supervised-data diagnostics used by
`FunctionalSolver.solve(...)` logging:

- `data_accuracy`
- `data_relative_l2_error`
- `data_rmse`

## Discrete point constraints

::: phydrax.constraints.PointSetConstraint
    options:
        members:
            - __init__
            - from_points
            - from_operator
            - data_metrics
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
