# Core constraints

For the mathematical conventions used by sampled constraints (residual norms, reduction modes,
measures, and `over=` semantics), see [Guides → Constraints and objectives](../../guides_constraints.md).

## Constraint containers

::: phydrax.constraints.FunctionalConstraint
    options:
        members:
            - __init__
            - from_operator
            - sample
            - loss

---

::: phydrax.constraints.PointSetConstraint
    options:
        members:
            - __init__
            - from_points
            - from_operator
            - loss

---

::: phydrax.constraints.IntegralEqualityConstraint
    options:
        members:
            - __init__
            - loss
