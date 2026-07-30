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

## Metric cochain residuals

`CochainResidualConstraint` samples one `CochainCells` degree from a fixed,
dataset, or graph-trajectory domain. Its residual must return a declared
cochain `DomainFunction` of that same degree. `from_program` binds a
`CochainResidualProgram`, so fixed-complex and operator-level topological
physics can share one residual definition.

Reduction is segmented by graph before cases are averaged:
`graph_mean` uses an arithmetic cell mean, `metric_mean` normalizes by
Hodge-star mass, and `metric_sum` retains physical Hodge-star mass. Static
padding is masked. For graph trajectories, temporal quadrature multiplies the
cochain measure rather than replacing it.

::: phydrax.constraints.CochainResidualConstraint
    options:
        members:
            - __init__
            - from_operator
            - from_program
            - sample
            - loss

---

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
