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

### Periodic seam projection

`prepare_periodic_projection(functions, conditions, route=...)` prepares every
`phydrax.conditions.Periodic` declaration of the program as one realization and
exposes its joint typed `condition`; compile it with
`EnforcementSpec(prepared.condition, realization=prepared)`. The route is
explicit and never falls back:

- `"analytic"` lifts the seam residual with the centered Bernoulli endpoint
  basis of each identified coordinate. The endpoint system of the declared jets,
  transports, and targets is prepared once through the native constraint operator;
  linearly dependent declarations, jet orders, rows, axes, and endpoint
  evaluations above `PeriodicResourcePolicy` are refused at preparation. Several
  coordinates of one field compose sequentially, which is exact when their event
  transports commute and constant targets agree at seam intersections; other
  compositions are refused. The pairing must be `PeriodicIdentification.pairing()`.
- `"coefficient"` assembles the joint condition on an explicit
  `AbstractLinearRepresentation` and eliminates it with `CoefficientElimination`.
- `"construction"` admits fields whose model carries a matching
  `PeriodicInputCertificate` (see [Embeddings](nn/embeddings.md)) and leaves them
  unchanged; it covers homogeneous identity-transport seams only and re-checks the
  certificate of the fields it realizes.

`PeriodicProjectionEvidence` reports the route, equality scope (`continuum`,
`finite-representation`, or `structural`), constituent condition identities,
per-coordinate `PeriodicAxisEvidence` (endpoint rows, rank, basis size, condition
number, maximum jet order), the derivative regularity each field must have, the
endpoint evaluations per query, and the `PeriodicPreservationRecord` of every
earlier wall or initial contract bound by the compiler. Only `"certified"` records
enter the admission's `preserves`; `"probed"` records are observations. `seam_defect`
samples the seam residual; it is diagnostic evidence, not a proof.

Endpoint work counts each declared source and target jet, not just distinct
geometric images: the bound is `product(1 + jets_per_axis)` per field before
compiler reuse. Two axes with value and slope requests require a bound of 25;
three require 125. Explicit tighter capacities refuse before numerical preparation.
Value-only preparation permits `maximum_order=0`.

Conditions on the same actual coordinate are fused even when identifications
have different diagnostic IDs. Geometric revisions and fixed targets participate
in prepared identity; source support and event layout must agree at preparation
and realization. Rejected attempts retain accepted lifecycle coordinates and stamps.
Construction admission checks the current native evaluator's evidence and input
packing, not just metadata copied from an earlier model.

For a local operator `B u = g` and affine seam target `J u = h`, compatibility
requires `J g = B h`. Normal/time derivative walls therefore annihilate a
constant `h`, while a Robin wall includes its value coefficient. The compiler
checks this complete operator relation rather than equating `J g` with `h`.

Polynomial and right-inverse casts must remain representable at the actual field
precision. Invalid casts or nonfinite lift values refuse during lazy field
evaluation; this is not a global finiteness certificate for an arbitrary field.


Every `AbstractFieldRealization` publishes a `RealizationAdmission` listing the
fields it reads, the fields it may write, the conditions it establishes, and the
conditions it preserves; a correction chart that may replace any field it
receives sets `writes_unknown` and counts as writing every field. The compiler
refuses a typed realization that may write a periodically enforced field without
preserving its seams, interior anchors on a periodic field, and walls that select
an identified face. The analytic route also refuses certified exact-PDE trial
fields, whose trial space the lift would leave.

::: phydrax.enforcement.prepare_periodic_projection

---

::: phydrax.enforcement.PreparedPeriodicProjection
    options:
        members:
            - realize
            - admission
            - seam_defect

---

::: phydrax.enforcement.PeriodicResourcePolicy

---

::: phydrax.enforcement.PeriodicProjectionEvidence

---

::: phydrax.enforcement.PeriodicAxisEvidence

---

::: phydrax.enforcement.PeriodicPreservationRecord

---

::: phydrax.enforcement.RealizationAdmission

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
