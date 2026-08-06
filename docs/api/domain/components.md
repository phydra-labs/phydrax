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

::: phydrax.domain.ComponentSpec
    options:
        members:
            - __init__
            - component_for

## Domain components

`DomainComponentUnion` is an additive collection of measure-disjoint
components, not an arbitrary geometric set union. It requires at least one
term, rejects duplicate terms, and requires every term to use the same
compatible labeled domain. Measures and sample allocations add across terms;
callers remain responsible for ensuring that separately filtered terms do not
overlap.

::: phydrax.domain.DomainComponent
    options:
        members:
            - __init__
            - measure
            - sample
            - sample_coord_separable
            - normals
            - normal
            - sdf
            - enforcement_gate

---

::: phydrax.domain.DomainComponentUnion
    options:
        members:
            - __init__
            - measure
            - sample


## Metric volume

`with_riemannian_measure` multiplies a component's existing integration weight
by `sqrt(det(g))` for one labeled coordinate factor. Sampling locations,
filters, masks, and all other product-measure factors remain unchanged.

::: phydrax.domain.with_riemannian_measure