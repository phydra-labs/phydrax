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

## Typed condition realizations

`phydrax.conditions` owns the condition operator, codomain, relation, and
quantifier. `phydrax.enforcement` owns deterministic ways to realize that
declaration. The principal realization families are:

- `ExactAffineProjector` for joint finite or fiberwise linear equalities;
- `CoefficientElimination` for certified finite linear representations;
- `LocalNonlinearRetraction` and `MinimumDistanceRetraction` for nonlinear
  equalities;
- closed-set projections and open-set `FeasibleParameterization` values for
  inequalities, cones, complementarity, and positivity.

These contracts are intentionally distinct. A local ansatz is not called an
idempotent projector, a sampled condition is not called continuum-exact, and a
probabilistic observation is handled by `phydrax.uq` rather than by deterministic
field realization.

### Joint affine projection

`prepare_affine_projector` assembles every field/condition block before preparing
one right inverse. It therefore supports cyclic coupled-field equations without a
pivot. `ConstraintLinearCorrectionProvider` uses the native linalg constraint
operator; kernel, geometry, cardinal, graph, interface, and represented-coefficient
providers implement the same correction contract.

The prepared projector exposes rank, nullity, right-inverse/range defects,
numeric-version, provider, and exactness-scope evidence. Factorization happens
during preparation or explicit refresh, never during field queries.

### Realization lifecycle

`EnforcementProgram.prepare_step` creates an all-or-nothing transaction over
fixed, caller, per-step, adaptive, parameterized, or randomized realization
sources. `EnforcementState` is checkpointable. Failed refreshes or realizations
withhold candidate fields; `commit_enforcement_step` commits only a successful
accepted-step transaction.

### Linear representations

`AbstractLinearRepresentation` exposes explicit coefficient extraction,
replacement, synthesis, and condition assembly. `CoefficientElimination` lowers
hard equalities into a native `ConstraintMap` plus a dynamic affine lift while
preserving representation certificates. The callable and substrate adapter
constructors require explicit actions; they never inspect arbitrary model
attributes.

### Geometry and RKHS providers

Boundary covers retain patch, junction, orientation, collar, represented-geometry,
and physical-geometry evidence. Trace providers distinguish analytic, represented,
and realized-discrete right inverses. Kernel providers distinguish canonical
minimum-RKHS corrections, exact finite-feature routes, selected-section
realizations, and tolerance-terminated matrix-free approximations. Exact kernel
preparation never adds hidden jitter.

Arbitrary Python callables receive no linearity or exactness certificate. They can
participate in soft or nonlinear evaluation, but exact affine preparation rejects
them unless a typed provider supplies the complete action and evidence.
