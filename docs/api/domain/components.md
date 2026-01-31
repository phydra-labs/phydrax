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

---

::: phydrax.domain.DomainComponentUnion
    options:
        members:
            - __init__
            - measure
            - sample
