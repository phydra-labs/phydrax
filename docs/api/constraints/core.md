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
            - data_metrics
            - loss

## Adaptive collocation

Adaptive policies are attached to `FunctionalConstraint.collocation_policy`.
See [Guides → Solvers and training → Adaptive collocation policies](../../guides_solver.md#adaptive-collocation-policies)
for estimator semantics, geometry support, and budget comparisons.

::: phydrax.constraints.CollocationPolicy
    options:
        members:
            - __init__
            - initialize
            - refresh
            - data_metrics

---


::: phydrax.constraints.SeparableCollocationPolicy
    options:
        members:
            - __init__
            - initialize
            - refresh
            - data_metrics

---

::: phydrax.constraints.HierarchicalAxisPolicy
    options:
        members:
            - __init__
            - initialize
            - refresh
            - data_metrics

---


::: phydrax.constraints.ControlledCollocationPolicy
    options:
        members:
            - __init__
            - initialize
            - refresh
            - settle
            - data_metrics

---

::: phydrax.constraints.RefreshSchedule
    options:
        members:
            - __init__

---

::: phydrax.constraints.ResidualMonitor
    options:
        members:
            - __init__

---

::: phydrax.constraints.RefreshGuard
    options:
        members:
            - __init__

---

::: phydrax.constraints.AdaptationBudget
    options:
        members:
            - __init__

---

::: phydrax.constraints.CoverageAnchors
    options:
        members:
            - __init__

---

::: phydrax.constraints.collocation_policy_support
    options:
        members: []

---


::: phydrax.constraints.PointSetConstraint
    options:
        members:
            - __init__
            - from_points
            - from_operator
            - data_metrics
            - loss

---

::: phydrax.constraints.IntegralEqualityConstraint
    options:
        members:
            - __init__
            - loss
