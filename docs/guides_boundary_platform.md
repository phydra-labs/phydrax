# Boundary platform qualification

The boundary platform separates **implemented code**, **bounded execution**,
**numerical evidence**, and **commercial support**. None of those labels implies the
next one. In particular, a finite result or a small discrete residual is not a
continuum error certificate.

## Exact support tuple

A support declaration applies to one exact tuple:

```text
geometry × trace × PDE/formulation × provider × precision × differentiation × platform
```

`BoundarySupportEnvelope` records a content ID for every coordinate. A declaration for
triangle DP0 traces does not cover continuous traces; a direct CPU provider does not
cover an FMM or GPU provider; and a primal action does not cover a differentiated solve.
Changing any coordinate produces a different `envelope_id`.

An envelope also carries a finite claim set. Unsupported claims are members of that set
with an explicit reason, rather than absent entries in a permissive capability lookup.
Evidence for a claim outside the set is rejected. Stop-ship conditions are part of the
fingerprint. There is no provider or capability registry and no fallback from an unknown
tuple to a nearby tuple.

```python
support = phx.operators.BoundarySupportEnvelope(
    geometry_id="closed-oriented-triangle-mesh",
    trace_id="triangle-dp0",
    pde_formulation_id="laplace-single-layer-dirichlet",
    provider_id="direct-blocked-reference",
    precision_id="float64-accumulate-float64",
    differentiation_id="none",
    platform_id="cpu-posix",
    claims=("finite-execution", "operator-action", "continuum-error"),
    unsupported_claims={
        "continuum-error": "No continuum discretization estimator is implemented."
    },
    stop_ship_conditions=("resource-preflight-failed",),
)
```

Claims, parents, artifacts, and stop-ship lists are bounded, duplicate-free, and
canonicalized before hashing. IDs and reasons are nonempty and bounded in length.
Instances are immutable Equinox `StrictModule` values.

## Evidence ladder

`BoundaryQualificationEvidence` uses five cumulative levels:

| Level | Meaning |
| --- | --- |
| `computed` | The declared provider produced a bounded record, or explicitly reported the claim as unsupported. |
| `checked-discrete` | A discrete identity, residual, parity check, or benchmark bound was checked against prerequisite computed evidence. |
| `quadrature-supported` | Quadrature or local integration error is bounded for the declared discretization and provider. |
| `continuum-qualified` | Discretization and continuum error are bounded over the declared support envelope. |
| `continuum-certified` | Independent certification evidence is linked to prior qualification evidence for that same envelope. |

Every level above `computed` requires a prerequisite evidence ID and a finite,
nonnegative error bound with a named metric. This makes an escalation without lineage
invalid. A `continuum-certified` label is mathematical/product qualification metadata;
it is not a legal, regulatory, or safety approval.

Unsupported is a separate state, not an error value. Unsupported evidence must use the
`computed` level, identify an artifact containing the fail-closed declaration, and carry
an explicit reason. It has no error metric or bound. Consequently, an unsupported claim
cannot be encoded as error `0.0`, and a genuine supported zero error remains
representable.

Evidence links both a `BoundaryProductProvenance.provenance_id` and a
`BoundaryOperationalEvidence.operational_id`. Empty lineage IDs are rejected.

## Q0--Q3 maturity is orthogonal

Maturity does not alias the evidence ladder:

| Maturity | Product meaning |
| --- | --- |
| `Q0` | Experimental, bounded exploration; no supported production use. |
| `Q1` | Repeatable engineering evaluation within a named envelope. |
| `Q2` | Qualified product use within a named envelope and operating procedure. |
| `Q3` | Release-gated commercial support with maintained evidence and stop-ship handling. |

For example, a Q1 implementation may have strong quadrature evidence but no continuum
qualification. Conversely, a research continuum argument does not establish Q2 or Q3
operational maturity. Constructors deliberately do not infer one axis from the other.

## Provenance and licensing

`BoundaryProductProvenance` records product, producer, provider, source-content,
license, clean-room record, and parent product/plan/result IDs. `source_kind` is one of
`native`, `clean-room`, `adapted`, or `external`. These fields preserve traceability and
license awareness; `clean_room_record_id` identifies the applicable engineering record
and does **not** assert legal approval. Legal review remains an external organizational
process and must not be invented in a qualification artifact.

Source content and licenses are identified, not embedded by this contract. A provider
change requires new provenance even if numerical inputs are unchanged.

## Fail-closed provider operation

`BoundaryOperationalEvidence` binds a plan and optional result to exactly one provider.
It records parent plan/result IDs, whether the provider is deterministic, security and
resource preflight evidence IDs, a positive byte limit, forecast bytes, observed bytes,
and stop-ship reasons.

Provider dispatch is fail closed:

1. Resolve an exact provider ID; an unknown provider is unsupported.
2. Complete the security preflight before execution. Provider identity, trusted code or
   binary content, input paths, and external-data provenance belong in that preflight
   artifact. The contract does not grant sandbox or legal approval.
3. Compute the resource forecast before allocation. A passing preflight cannot forecast
   more than the declared byte limit.
4. Execute only after both preflights pass. A failed preflight cannot have a result ID.
5. Record observed bytes only with a result ID. An observation above the limit remains
   recordable only with an explicit stop-ship reason; it is not silently accepted.

A deterministic provider omits a nondeterminism reason. A nondeterministic provider must
name one. Determinism is descriptive evidence, not inferred from provider branding.
The contracts add no telemetry and make no network calls.

## Persistence: manifests, not pickle

Do not pickle boundary qualification objects. Pickle can execute code while loading,
does not provide a stable cross-release contract, and can bypass constructor validation.
Persist a primitive manifest containing the constructor fields plus the resulting
content IDs, and store numerical arrays or reports in separately content-addressed
artifacts. On load:

1. verify artifact content IDs and provenance/license IDs;
2. reconstruct each object through its public constructor;
3. compare the reconstructed fingerprint with the stored content ID;
4. reject unknown fields, unknown evidence levels, unknown claims, and unresolved
   provider IDs.

A stored fingerprint authenticates canonical metadata identity, not the truth of an
unverified external report. Signing, access control, and artifact retention are outside
these value types.

## Derivative support

Differentiation is one coordinate of the support tuple, and each prepared owner states
it explicitly. The prepared product and each of its results carry an
`OwnerDerivativeCapability`. It lists the runtime arguments that may be
differentiated, the `DerivativeSurface` each one enters, and the route. It also lists
every refused quantity with a reason. The linear policy's `DifferentiationPolicy`
selects the route at preparation. Geometry, matching, panel pairing, and quadrature
are prepared once on the host and are never traced.

| Prepared owner | Mode | Admitted arguments (surface, route) | Refused |
| --- | --- | --- | --- |
| `prepare_scalar_laplace_fem_bem_3d` | `none` (default) | none (stopped) | all runtime arguments; fixed structure |
| | `rhs-only` | `volume_source_coefficients`, `dirichlet_jump`, `conormal_jump` (solver argument, implicit) | `conductivity`; `geometry`, `kernel`, `quadrature`, `exterior_conductivity` |
| | `mathematical` | the three data arguments above plus per-cell `conductivity` (physical parameter, implicit) | `geometry`, `kernel`, `quadrature`, `exterior_conductivity` |
| | `algorithmic` | refused at preparation | — |
| `prepare_elasticity_fem_bem_3d` | `none` (default) | none (stopped) | both loads; fixed structure |
| | `rhs-only` | `interior_load`, `boundary_load` (solver argument, implicit) | `geometry`, `kernel` (Kelvin/Lamé), `quadrature`, `interior_operator` (caller `A_sym` including the hypersingular term), `trace_maps` (`C`, `C^T`) |
| | `mathematical`, `algorithmic` | refused at preparation | — |
| `prepare_scalar_laplace_galerkin_2d` | linear actions | `dirichlet`, `conormal`, `far_field_constant` (input, direct) | `geometry`, `kernel`, `quadrature`, field `targets` |
| `prepare_exterior_laplace_dirichlet_2d` | `rhs-only` (default) | `dirichlet` (solver argument, implicit) | `geometry`, `kernel`, `quadrature` |
| | `none` | none (stopped) | `dirichlet`; fixed structure |
| | `mathematical`, `algorithmic` | refused at preparation | — |

The scalar conductivity route is exact for the affine P1 envelope. The interior
stiffness is `G^T diag(κ|T|) G`, where the cell-gradient map `G` is prepared once. A
runtime conductivity changes only the diagonal and rebinds the prepared solve through
`phydrax.linalg.refresh`, with no replanning. A compiled gradient therefore runs again
for a new conductivity without host synchronization. The transmission jumps are DP0
data, `g_D = γ0⁻u − γ0⁺u` and `g_N = κγ1⁻u − γ1⁺u`, with the same interior-to-exterior
normal as the conormal unknown.

Evaluation semantics:

- If a JVP, VJP, or gradient requests an argument that the capability does not admit,
  it raises `ValueError` beginning with `derivative-unsupported` at transformation
  time. A stopped route never returns a silent zero.
- Primal acceptance (`valid` or `accepted`) and `derivative_valid` are separate
  evidence. `derivative_valid` also requires an admitted route and a converged solve.
  The derivatives of a result that is not accepted are NaN under the status failure
  mode and raise under the error failure mode. Examples are a failed or non-converged
  solve, a nonpositive conductivity, a failed recertification, and a violated decaying
  far field.
- Implicit tangent and adjoint solves follow the policy's `LinearDerivativeSolvePolicy`.
  A derivative solve that misses its residual contract returns NaN rather than an
  approximate derivative.
- A derivative study of a two-dimensional `decaying` exterior solve must perturb inside
  that compatibility regime. A perturbation whose far-field constant exceeds the
  tolerance is not accepted, so its derivative is NaN.

## Current and planned support

The current implementation boundary must not be read as a qualification matrix:

| Slice | Availability | Qualification statement |
| --- | --- | --- |
| PR208 3D Laplace single-layer triangle-DP0 Galerkin and capacitance path | Implemented with explicit geometry and memory bounds | **Bounded/experimental.** It reports assembly, solve, quadrature, and resource evidence, but explicitly has no continuum discretization-error estimator. It remains Q0 until envelope-specific evidence promotes it. |
| 2D Laplace P1/DP0 Galerkin `V`/`K` and bordered exterior Dirichlet-to-Neumann solve on closed straight-panel polygons | Implemented with blocked-direct actions, pair-class quadrature evidence, and explicit resource budgets | **Bounded/experimental.** `ScalarLaplaceGalerkinReport2D.support` is its exact candidate `BoundarySupportEnvelope`; continuum error, open, curved, or moving geometry, Helmholtz, hypersingular, FMM Galerkin, geometry-derivative, and field-target-derivative claims are declared unsupported; fixed-geometry density and Dirichlet-data derivatives are claimed (see [Derivative support](#derivative-support)). It remains Q0. |
| Existing 2D/3D direct, adaptive, QBX, treecode, and FMM layer-potential paths | Implemented only for the contracts documented by each API | **Bounded/experimental for commercial qualification.** Existing certificates and evaluator reports do not automatically constitute continuum qualification. |

The scalar Calderón/trace, scalar transmission and open-screen, scalar and
elasticity FEM--BEM, finite-depth potential-flow hydrodynamics, hydroelastic response,
elasticity/Stokes, RWG Maxwell, scalar and Maxwell-field periodic, convolution-
quadrature, checked-linear-algebra, SurfaceModel/high-order/interchange, and
fast/adaptive/archive slices are implemented on this branch. They remain
**bounded/experimental** until each exact tuple has Q1--Q3 qualification evidence.
Their presence in an API is not a broad support claim, and no provider, formulation,
differentiation mode, accelerator, geometry format, or platform should be inferred
beyond the explicit envelope returned by that prepared product.

Wave C adds exact prepared-near Laplace DP0 `laplace-fmm-3d`,
`scalar-h-matrix-3d`, and `scalar-h2-matrix-3d` actions. Their policies bound
tree depth, blocks, local rank/order, resident bytes, and measured block error;
transpose/adjoint reverse stored factors, and no global dense matrix is formed.

Scalar closed-surface calculus now pairs DP0 Neumann traces with continuous-P1
Dirichlet traces. The hypersingular operator is the Maue-regularized P1 map;
DP0 hypersingular requests remain mathematically invalid and are not projected.

Boundary trace spaces are published to coupling consumers as
`BoundaryTraceSpaceCapability` and `CauchyTraceCapability` records (2-D polygon and
3-D closed-surface P1 Dirichlet / DP0 Neumann data, RWG currents, and their
Buffa--Christiansen duals). Each record states the trace quantity, representation,
conformity, orientation relative to the declared interior or the oriented surface,
physical Gram pairing, and geometry revision. The record is an identity and pairing
contract only: it adds no volume support, no qualification claim, and no route between
scalar Cauchy data and tangential currents.

`prepare_periodic_maxwell_boundary_3d` combines the prepared central
free-space RWG action with explicitly bounded smooth noncentral Bloch images.
Its evidence states that the finite image sum is not an infinite-lattice/Ewald
certificate. The same finite-versus-infinite distinction applies to the
rank-two periodic free-surface Green product.

Dynamic vector coupling uses the existing CQ controller through
`prepare_dynamic_elasticity_fem_bem_cq_3d` and
`prepare_dynamic_maxwell_fem_bem_cq_3d`; every complex node must be a prepared
native FEM--BEM solve with matching forward/transpose/adjoint families, and one
failed node invalidates the complete history.

Nonmatching scalar and Maxwell couplings consume explicit common-refinement
cross masses. Preparation requires complete coverage, positive orientation,
geometric residual evidence, full inf-sup rank, exact trace/load transpose,
and (for Maxwell) a bounded commuting defect. Screen junctions require a
declared continuity/flux law and a full-row-rank constraint saddle system.

Open-sheet displacement discontinuity uses continuous vector-P1 jumps and a
regularized isotropic elastic hypersingular action; constant DP0 jumps remain
unsupported. `BEMFractureProblem3D` composes that traction operator with
unilateral, cohesive, and friction-cone projections, and growth requires
explicit complete parent routes. Active-set and growth transitions are
nondifferentiable.

Promotion is per tuple and per claim. There is no family-wide promotion from one mesh,
precision, benchmark, provider, or platform.
