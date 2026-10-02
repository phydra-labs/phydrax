# Boundary and initial conditions

These classes state scientific semantics. Use `ResidualPenalty` for soft
realization or `EnforcementSpec` for an exact transform.

::: phydrax.conditions.Dirichlet
    options:
        members:
            - __init__
            - residual

---

::: phydrax.conditions.Neumann
    options:
        members:
            - __init__
            - residual

---

::: phydrax.conditions.Robin
    options:
        members:
            - __init__
            - residual

---

::: phydrax.conditions.Absorbing
    options:
        members:
            - __init__
            - residual

---

::: phydrax.conditions.Initial
    options:
        members:
            - __init__
            - residual

## Periodic seams

`Periodic` states one transported trace relation across a periodic seam,

```text
T[u_target](upper) - Gamma S[u_source](lower) = g,
```

where the lower face of the `PeriodicIdentification` is always the source and the
upper face the target. `S` and `T` default to the `order`-th derivative along the
identified coordinate; `trace_actions=(source_action, target_action)` replaces
them with a certified constant-coefficient `JetAction` pair, for example a
constitutive flux `JetAction({1: k})`. `transport=1` is ordinary periodicity,
`-1` antiperiodicity, a unit complex phase `exp(i k L)` Bloch matching, and an
`EventLinearMap` transports vector events. A nonzero `target` is an affine jump,
such as `-dp` for a pressure drop `p(upper) - p(lower) = -dp`.

One field name binds a same-field seam once; a `(source_field, target_field)`
tuple couples two distinct fields in that canonical order. Each declaration owns
exactly one order: matching values and first derivatives is two declarations.

The same object is a soft residual (`ResidualPenalty`), lowers through
`as_condition()` to a certified linear `PeriodicTraceAction` with a separate
`Equality(target)`, and is enforced exactly through
`phydrax.enforcement.prepare_periodic_projection`.

::: phydrax.conditions.Periodic
    options:
        members:
            - __init__
            - residual
            - action
            - as_condition

---

::: phydrax.conditions.JetAction
    options:
        members:
            - __init__
            - derivative

---

::: phydrax.conditions.PeriodicTraceAction
