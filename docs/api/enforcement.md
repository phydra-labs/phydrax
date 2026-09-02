# Exact enforcement

Enforcement compiles declarative conditions into exact field transforms. A
condition can therefore be realized softly with a penalty or exactly with an
`EnforcementSpec` without changing its scientific meaning.

::: phydrax.enforcement.EnforcementSpec
    options:
        members:
            - __init__
            - apply

---

::: phydrax.enforcement.EnforcementOptions
    options:
        members:
            - __init__

---

::: phydrax.enforcement.compile

## Low-level transforms

The compiler selects these transforms from the condition type. They are also
available for custom ansatz construction.

::: phydrax.enforcement.enforce_dirichlet

---

::: phydrax.enforcement.enforce_neumann

---

::: phydrax.enforcement.enforce_robin

---

::: phydrax.enforcement.enforce_initial

---

::: phydrax.enforcement.enforce_sommerfeld

---

::: phydrax.enforcement.enforce_traction

---

::: phydrax.enforcement.enforce_blend

---

::: phydrax.enforcement.enforce_graph_values

---

::: phydrax.enforcement.enforce_cochain_values

See [Solver exact enforcement](solver/enforcement.md) for staging and
a complete solver example.

## Typed affine trace equations

Hard conditions are finite affine equations built from `field_jet` and `equal`.
Each `AffineEnforcementTransform` pairs the equation with a typed
`TraceLifting` and checked `EnforcementProofObligations`. For example, a
two-field value relation is represented as
`equal(2 * field_jet("u", "x") + field_jet("v", "x"), target)` with one
declared pivot field. The compiler checks field/support alignment, the unique
pivot, derivative requirements, dependency cycles, and provider evidence
before producing the existing `EnforcementProgram`.

Arbitrary Python callables, nonlinear products of unknown traces, singular
pivots, and sampled equalities are not enforcement proofs and are rejected.
Normal jets require geometry normal capability. Preservation and topology
selection are preparation-time obligations; derivatives are supported only
inside the fixed compiled program.
