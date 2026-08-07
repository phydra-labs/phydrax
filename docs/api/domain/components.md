# Components

Components select which subset of a domain is being sampled (interior, boundary,
fixed-time slices, etc.) and wrap these into `DomainComponent` objects.

## Component markers

::: phydrax.domain.Interior
    options:
        members:
            - __init__

---

::: phydrax.domain.Boundary
    options:
        members:
            - __init__

---

::: phydrax.domain.Fixed
    options:
        members:
            - __init__

---

::: phydrax.domain.FixedStart
    options:
        members:
            - __init__

---

::: phydrax.domain.FixedEnd
    options:
        members:
            - __init__

---

::: phydrax.domain.SelectionSpec
    options:
        members:
            - __init__
            - selection_for

## Bound components

`Domain.component(...)` binds one selection per complete joint factor and records
that factor's base measure. Unrestricted components carry an `ExactMass` when the
factor can certify it. Predicate restrictions and unnormalized densities carry an
`UnknownMass` until a numerical estimator supplies evidence.

`ComponentSum` is an additive collection of measure-disjoint components, not a
geometric Boolean union. Terms must share the same support; duplicates are
rejected. Predicate-transformed terms require `assume_disjoint=True` because
overlap cannot be proven structurally.

::: phydrax.domain.DomainComponent
    options:
        members:
            - __init__
            - factor_component
            - base_measure
            - mass
            - restrict
            - with_density
            - sample
            - normals
            - normal
            - sdf
            - enforcement_gate

---

::: phydrax.domain.ComponentSum
    options:
        members:
            - __init__
            - base_measure
            - mass
            - sample

## Measure contracts

::: phydrax.domain.BaseMeasure

::: phydrax.domain.ExactMass

::: phydrax.domain.EstimatedMass

::: phydrax.domain.UnknownMass

## Metric volume

`with_riemannian_measure` multiplies a component's existing integration weight
by `sqrt(det(g))` for one labeled coordinate factor. Sampling locations,
filters, masks, and all other product-measure factors remain unchanged.

::: phydrax.domain.with_riemannian_measure