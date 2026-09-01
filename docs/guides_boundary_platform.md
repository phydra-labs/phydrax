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

## Current and planned support

The current implementation boundary must not be read as a qualification matrix:

| Slice | Availability | Qualification statement |
| --- | --- | --- |
| PR208 3D Laplace single-layer triangle-DP0 Galerkin and capacitance path | Implemented with explicit geometry and memory bounds | **Bounded/experimental.** It reports assembly, solve, quadrature, and resource evidence, but explicitly has no continuum discretization-error estimator. It remains Q0 until envelope-specific evidence promotes it. |
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

Promotion is per tuple and per claim. There is no family-wide promotion from one mesh,
precision, benchmark, provider, or platform.
