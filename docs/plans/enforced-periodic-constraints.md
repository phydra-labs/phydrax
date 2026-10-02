# Enforced periodic constraints: end-to-end implementation plan

## Implementation disposition

This file preserves the original design proposal; the current public contract is
documented in `docs/guides_conditions.md` and `docs/api/enforcement.md`. The shipped
Cartesian analytic route uses same-field self-seams and `JetAction` trace pairs.
Interior anchor overlays and unsupported overlapping hard realizations are
explicitly refused, not silently repaired or advertised as jointly enforced.
Functional local-data compatibility requires certification by default, with
sampled compatibility available only through explicit `data_compatibility="probed"`.
Input-kind checking follows the current `phydrax.typing.checked` owner boundary,
while scientific, support, numerical, and source-identity checks remain explicit.
The prospective file map and completion matrix below are design history, not a
claim that every originally proposed realization/provider combination is supported.

## 1. Execution contract and outcome

This document is a plan, not an implementation or a runtime qualification claim. Repository research was read-only; no tests, benchmarks, package code, or examples were executed. Line references describe the inspected snapshot and must be refreshed before implementation.

A later `//code` implementation must use a fresh worktree from the then-current branch under `/Users/lgleyzer/PHYDRA/phydra-labs/.worktrees/`, copy the authoritative ignored `AGENTS.md` into it before work, and retain all implementation changes there. `//code--` is the explicit exception. Preserve unrelated concurrent edits. No commits, merge, or PR are implied by this plan.

The deliverable is a first-class periodic field condition usable both as a soft residual and as an enforced constraint. It must support ordinary periodic values and declared coordinate jets, same-field and same-patch seams, scalar antiperiodic/Bloch transport, affine jumps, multiple Cartesian periodic directions, a certified finite-coefficient route, and explicit periodic model construction. Compatible boundary, initial, and finite observation constraints must remain satisfied. Unsupported geometry, unknown preservation, incompatible targets, insufficient regularity, rank loss, and resource overflow must be reported, never hidden by wrapping, approximation, jitter, or zero replacement.

The complete implementation includes the declaration, prepared realization, compiler/solver integration, topology adapters, numerical and differentiation evidence, failure/lifecycle behavior, tests, benchmarks, examples, documentation, and regenerated public/capability data. Completing only a periodic feature layer or matching collocation endpoints is not completion.

## 2. Baseline facts and corrections

| Inspected owner | Existing contract | Consequence |
| --- | --- | --- |
| `conditions/_base.py:71-105`, `terms/_residual.py:300-352` | `ResidualPenalty` accepts an `AbstractResidualCondition` whose residual is one `DomainFunction`. | Keep one public scientific declaration per residual trace; do not widen all soft terms to arbitrary typed product conditions. |
| `conditions/_lowering.py:301-347` | Generic residual lowering installs an uncertified legacy operator. | Add specialized native lowering; arbitrary residual callables remain uncertified. |
| `conditions/_ir.py:111-126,230-270` | Field codomains and unique external sources already exist. Field codomain validation currently checks support, not evaluated fiber shape/dtype. | A same-field seam binds one source once. The periodic action owns evaluated fiber validation. |
| `conditions/_subdomain.py:42-51` | Existing subdomain jumps require distinct field bindings. | Do not invent two aliases for one field or weaken every ordinary interface contract. |
| `domain/decomposition/_cover.py:146-288` | `PairedSupport` owns paired coordinate maps, but forbids equal patch IDs. Its periodic audit skips ambient map agreement. | Admit self-pairing only with an explicit periodic identification; audit the identification instead of claiming that skipped checks prove it. |
| `domain/decomposition/_cartesian.py:807-831` | A periodic wrap pairing is generated only when the partition count exceeds one. | Add a real single-patch seam; preserve existing coordinate-wrap semantics rather than diagnosing all one-patch evaluation as broken. |
| `enforcement/_spec.py:379-413` | Typed conditions can carry an `AbstractFieldRealization` outside the one-pivot local ansatz path. | Periodic projection belongs in the typed realization path, not a new local `TraceLifting` kind. |
| `enforcement/_fiber.py:406-489,725-794` | Analytic fiber units implement action/target/right-inverse lift. They are not lifecycle field realizations. Shared writes must be fused. | Add the missing realization integration and proof, not a second correction engine. |
| `enforcement/_compile.py:2019-2066` | Local overlays precede sequential typed realizations; atomic failure does not prove preservation of earlier constraints. | Add overlap/preservation admission; ordering alone is not sufficient. |
| `enforcement/_linear_representation.py` | Explicit extraction, replacement, synthesis, condition assembly, and coefficient elimination exist. | Reuse this route for complete represented trace constraints. |
| `discretization/spectral/_constraints.py` | Analytic one-sided endpoint rows exist; paired endpoint rows do not. | Extend the spectral owner; enforcement must not copy private endpoint formulas. |
| `enforcement/_polynomial_representation.py` | Owns sparse algebraic invariant/equivariant subspaces, not numerical endpoint polynomial bases. | Intentionally unchanged for this feature. |
| `nn/layers/_fourier_embeddings.py:269-307` | Fixed harmonics can construct periodic features, but the helper does not prohibit periodic-coordinate passthrough. | Strengthen the construction contract and carry truthful structural evidence. |
| `kernels/_transforms.py:25-110` | Positive-definite pullback kernels already exist. | Reuse a smooth periodic input transform; do not create a parallel periodic-kernel runtime. |
| `discretization/_axis_domain.py` | `AxisDomain.periodic` is already canonical for numerical axes. | Provide an explicit binding adapter; do not create unrelated period/axis flags. |

Language-server reference queries covered `PairedSupport`, `PairedSupportEvidence`, and `FiberProjectionState`. Pairing consumers include domain cover/geometry auditing and `solver/coupling/_interfaces.py`; the plan below includes them. Implementation must refresh references for every changed exported symbol, including symbols not yet selected for modification.

## 3. Frozen design decisions

### 3.1 One declaration, independent realizations

Add `phydrax.conditions.Periodic`, an `AbstractResidualCondition`, in `conditions/_periodic.py`.

- Bind either one source field or two explicitly distinct local field sources across a periodic seam. Same-field matching uses one source and two trace maps, never duplicate `FieldSpec.source` entries.
- Each declaration owns one requested trace relation. Value matching is the default; an explicitly requested derivative order means that order only, not an implicit promise about all lower or higher orders. Several orders are several declarations fused for hard enforcement.
- Use the existing derivative request/jet vocabulary. The common axis-aligned case resolves the differentiation variable/component from the identification; mixed coordinate requests and oriented normal/linear-flux requests retain their explicit owning declarations. Do not create a second string-based dimension system.
- Support a constant scalar transport and a target `DomainFunction` on the common seam. Identity transport is ordinary periodicity; minus one is antiperiodicity; a declared unit complex phase is Bloch matching; nonzero targets are affine jumps.
- Certified event transport and linear flux operators may use existing typed operator contracts. Raw callbacks remain available to generic soft conditions but cannot claim a certified periodic hard action.
- Targets and event codomains are explicit. Equal array lengths do not establish vector-space, axis, or transport identity.
- `residual(functions)` returns the transported difference minus the target on the paired support. `as_condition()` lowers the very same declaration to a certified action plus `Equality(target)`. The target is not subtracted inside the certified linear action.
- The periodic action certifies linearity only. It does not claim a function-space adjoint without an owning representation/metric provider.

The canonical public hard entry point is `phydrax.enforcement.prepare_periodic_projection(functions, conditions, ...)`. It prepares one joint realization and exposes its joint typed `condition` for the existing `EnforcementSpec(condition, realization=...)` interface. No `enforce_periodic` forwarding alias, renamed local ansatz family, or periodic-only solver API is added.

The prepared operation explicitly selects a supported analytic, represented-coefficient, or constructed-space realization. A route cannot silently fall back to sampled matching. Provider/resource policy is explicit; existing `AffineProjectionPolicy`, constraint preparation, regularity, and lifecycle policies are reused rather than duplicated.

### 3.2 Geometry owns identification; fields own matching

Add a canonical `PeriodicIdentification` in `domain/_periodic.py`.

It records the source support, geometric coordinate/component identity, opposing supports, paired coordinate maps, displacement/orientation, and period. Cartesian constructors derive endpoints from the owning factor and preserve the physical numeric representation. Relabeling must transform the explicit coordinate binding rather than reconstruct identity from display names.

Freeze geometric equation direction independently of cover traversal: the canonical source is the lower face and target is the upper face, so the field action is `upper_trace - transport * lower_trace`. Existing Cartesian pairings enumerate `left=upper`, `right=lower`; retain that traversal order and store the explicit source/target side roles. Never infer equation direction from left/right display names. Thus conventional Bloch transport is `exp(i*k*L)` and a pressure drop target is `-DeltaP` for `p(upper)-p(lower)=-DeltaP`.

`PairedSupport` remains the paired-trace carrier. It references the identification for periodic topology; ordinary shared interfaces retain their distinct-patch requirement. A Cartesian periodic cover constructs one identification and binds its seam to the actual endpoint patches, including the same patch when the axis is undivided.

A standalone identification constructs the actual full-domain source patch/common seam using the same domain-owned preparation; it does not create two fictional field aliases or an artificial second patch.

Provide an explicit physical-boundary view driven by the identification set. It returns the actual unpaired faces and their supports. A fully periodic interval has no physical boundary; represent that as an empty face collection, not a fabricated zero-measure boundary or placeholder geometry. Periodic seams remain available separately as trace supports.

The original interval/box remains a fundamental-domain representation. Its ordinary `Boundary()` is not silently redefined. Physical boundary selection, enforcement cover preparation, and numerical-axis binding consume the explicit identification set. This avoids both a broad quotient-domain wrapper and several independent `periodic=True` authorities.

### 3.3 Exactness is scoped and conditional on regularity

Keep separate evidence dimensions for:

1. source/geometry construction and identity;
2. equality scope: continuum trace, complete finite trial space, or finite realization;
3. available derivative regularity;
4. numerical residuals and tolerances;
5. preservation of other constraints and lifecycle/source revisions.

Structural construction and continuum analytic lifting both establish mathematical relations, but have different provenance. Sampled audits are observations, not a fourth continuum certificate. Do not add a competing enum if the existing exactness vocabulary and certificate provenance represent these distinctions; extend an existing owner only where a concrete missing distinction is required.

C-infinity periodic features do not make a nonsmooth downstream model C-infinity. A finite-order trace projection certifies only the requested jets and needs the corresponding interior regularity to claim a smooth periodic extension. Floating-point residuals are reported with declared precision/tolerance; structural exactness must not mean bitwise equality of independent sine/cosine evaluations.

### 3.4 Preserve or jointly enforce

For a constraint action A, later corrections must lie in its homogeneous space: A(delta u) = 0. Alternatively, all overlapping constraints must be solved in one joint realization.

- Separate periodic axes on one field are fused into one fiber unit. Axiswise internal execution is permitted only when the declared transport/targets and derivative actions establish commutation and corner compatibility.
- Existing boundary/initial/anchor overlays cannot simply be moved before or after periodic projection and assumed safe.
- Unknown compatibility is not invalid data: report unavailable exact evidence/unsupported composition separately from a demonstrated conflicting target.
- Finite probing may reject a mismatch; it cannot prove compatibility of arbitrary callable data.
- Preserve existing exact-trial-space refusal unless the representation owner certifies that the periodic transform stays in its trial space.

### 3.5 Proposed public interfaces and admission matrix

Freeze these interfaces before implementation; names below are proposed additions, not existing callable APIs.

| Public surface | Parameters and result | Owning invariant |
| --- | --- | --- |
| `domain.PeriodicIdentification` | Source `Domain`, explicit coordinate label/component, and optional explicit identification ID; source/target endpoint roles and period derive from that factor. | Validated Cartesian identification with semantic coordinate identity, real seam/patch preparation, and canonical revision. |
| `conditions.Periodic` | `fields: str | tuple[str, str]`, `pairing: PairedSupport`; keyword `order: int = 0`, `transport: ArrayLike | EventLinearMap = 1.0`, `target: DomainFunction | ArrayLike = 0.0`, explicit event `value: ArrayCodomain | None = None`, optional certified per-side `trace_actions`, and `label`. | One transported trace equality; unique source binding; order applies along the identification's declared coordinate. `value=None` means scalar, not inferred vector semantics. |
| `conditions.Periodic.as_condition` | Existing field/codomain/ID declaration parameters, validated against the periodic declaration. | Certified native lowering, source-domain codomains, one residual field codomain, and separate affine target. |
| `enforcement.prepare_periodic_projection` | Source fields and periodic declarations; explicit `route`, optional representation, existing affine policy, and bounded resource policy. Returns `PreparedPeriodicProjection`. | One jointly prepared realization with `.condition`, constituent condition IDs, bound source/geometry identities, evidence, refresh, and preservation requirements. |
| `PreparedPeriodicProjection.realize` | Existing `ConditionEvaluationContext` and lifecycle state contract. | Atomic results through `FieldRealizationResult`, not a second direct application API. |

Declare `PeriodicProjectionRoute` once using the runtime-importable selector convention, with analytic, coefficient, and construction choices. Parse through `phydrax.typing.parse` and use exhaustive dispatch. Do not add `auto` or approximate fallback. The coefficient route requires a representation; the construction route requires a bound full-field certificate; incompatible route arguments refuse.

Per-side custom trace actions must be a complete certified pair with declared input/output support and event spaces. Nonzero `order` cannot additionally re-differentiate custom trace actions. Raw callbacks are not accepted as certified trace actions. Coordinate value/jet matching is the implemented built-in; linear-flux matching is implemented through these native actions, with a tested constitutive example.

Tuple ordering is canonical `(source_field, target_field)`, not cover `(left_field, right_field)`. A single string binds that source to both roles. The per-side `trace_actions` type is `tuple[AbstractConditionOperator, AbstractConditionOperator] | None`, in the same `(source_action, target_action)` order; each action acts on its declared source domain before pullback. For Cartesian wraps this maps source to cover right/lower and target to cover left/upper. Bindings and custom actions follow those explicit roles even for asymmetric distinct-source fields.

`PeriodicResourcePolicy` owns actual maximum axis/jet/query/endpoint/observation capacities and retained/preparation bytes not already covered by a native policy. Use static nominal dimension/identifier contracts and explicit dtype forms for its prepared numerical leaves. It does not duplicate native solve tolerance/rank/precision policy.

Do not overload `EnforcementSpec(Periodic)` into the local ansatz path. The documented flow is: declare the common pairing and periodic traces; prepare one projection; pass `EnforcementSpec(prepared.condition, realization=prepared)` to the existing compiler. Soft treatment uses those same `Periodic` objects with the existing `ResidualPenalty`.

During compiler preparation, binding prior boundary/initial/anchor contracts may require a new immutable joint prepared artifact. Reprepare through the same owner once during compilation, not during runtime queries and not by mutating a supplied prepared object. Report the effective joint preparation identity to consumers.

## 4. Mathematics and numerical preparation

### 4.1 Ordinary endpoint jets

For one interval of length L and transverse coordinates y:

`J_k u(y) = partial_x^k u(b,y) - partial_x^k u(a,y)`.

An analytic lift can use

`p_k(x) = L^k B_(k+1)((x-a)/L)/(k+1)!`,

whose endpoint derivative jumps satisfy `J_j p_k = delta_jk`. Thus

`u_hat = u + sum_k p_k (g_k - J_k u)`.

Only declared rows are corrected. Endpoint corrections are functions of y; their x-dependence comes solely from the lift. Freeze the ordinary analytic route to the centered Bernoulli gauge, with zero mean correction polynomials. For value and first-derivative matching, its exact elementary reference is

`u_hat = u - (s-1/2) Delta_0 - (L/2)(s^2-s+1/6) Delta_1`, where `s=(x-a)/L`.

This is an independent low-order oracle for the same chosen projection. The uncentered `s` and `s^2-s` formulas also remove endpoint jumps but define a different interior correction; do not compare their interior outputs to the centered route as if the projectors were identical.

The projection is idempotent for fixed affine targets. It is not advertised as a minimum-function-norm correction. No complete coordinate Jacobian or Hessian is required: use the owning derivative request/jet actions and contracted derivatives.

### 4.2 Transported traces and fluxes

For constant scalar transport gamma, the canonical action is `upper_jet - gamma * lower_jet`, using identification source/target roles rather than cover traversal. Affine targets remain separate. General certified event/flux transport uses its explicit source/target spaces and oriented trace actions.

Do not assume that the ordinary Bernoulli lift is a right inverse for nonidentity transport. Prepare the transported endpoint action on a bounded polynomial correction basis and obtain its right inverse through native linalg. Include enough basis functions to retain surjectivity at identity and antiperiodic transport; test phase limits rather than divide by `1-gamma`.

Prepared geometry/transport is factored once and reused. A changed numeric transport or geometry requires explicit native refresh/rebinding and lifecycle invalidation, not refactorization at each query. Rank, conditioning, compatibility, and resource evidence reach the result.

For scalar Bloch model construction, an explicitly authored phase factor times a periodic remainder is valid. An explicit Bloch wavevector is scientific identity; do not infer it uniquely from a phase modulo two pi. For affine pressure drops, an explicitly declared affine lift plus a periodic remainder is valid.

Coordinate derivatives on opposite Cartesian faces have the same orientation. Outward normal derivatives have opposite orientations. Conserved flux matching also includes the owning constitutive operator; it is not blindly equated to derivative matching.

### 4.3 Multiple directions and intersections

Homogeneous Cartesian coordinate projectors can be composed when their actions/lifts commute. With affine targets, verify the corresponding transverse trace identities at corners; scalar transport phases commute but their prescribed jumps still need compatibility. General noncommuting event transports must use a jointly represented problem or be refused by the separable route, not declared inconsistent merely because scalar separability is unavailable.

Bound endpoint work explicitly in terms of the number of axes, declared jet orders, and query capacity. Avoid a recursively nested projector that expands to an uncontrolled Cartesian product of endpoint evaluations. Cache shared endpoint/edge/corner requests within a prepared evaluation using existing derivative planning/workset infrastructure. Chunk independent numerical lanes; use stable compiled callables, not per-call jit creation.

### 4.4 Finite trial spaces and redundant rows

The spectral/representation owner assembles complete analytic endpoint coefficient maps. In a tensor-product space, each transverse coefficient participates in the seam relation; finitely sampling transverse coordinates is not a substitute unless unisolvence and the complete reconstruction map are certified.

Use `LinearConditionAssembly` and `CoefficientElimination`. Preserve coefficient order, realification, PyTree structure, source IDs, and exact trial-space certificates. Omit Fourier endpoint rows only when the representation certificate covers the complete requested relation and target. Ordinary Fourier construction certifies identity-transport homogeneous matching, not arbitrary Bloch/antiperiodic or affine relations. Otherwise assemble the transported coefficient action; a nonzero affine jump on an unlifted ordinary Fourier space is unrealizable and must refuse with compatibility evidence, unless an explicit certified affine lift is supplied.

Multiple axes and corner intersections can produce dependent rows. Canonicalize identical declarations before preparation, reject conflicting duplicates, and retain every original condition identity. For genuine linear dependencies, use a rank-revealing reduction owned by native linalg with an explicit map from reduced constraints to the original system and full target-compatibility evidence. Do not silently drop rows. If exact equivalence is unavailable, the exact route refuses; a generalized/approximate result cannot satisfy `exact_required=True`.

## 5. File-by-file implementation

### P1. Identification and paired topology

**New `phydrax/domain/_periodic.py`**

- Implement `PeriodicIdentification`, Cartesian geometric preparation, canonical coordinate/component identities, opposing support maps, and explicit physical-boundary enumeration.
- Own validation for finite positive periods, correct source domains, declared coordinate components, and orientation. Constructors validate before assignment.
- Bind to `AxisDomain.periodic` through an explicit adapter with the same endpoint identity. Numerical dtype/precision choices stay with the discretization owner.
- Reuse `discretization/_periodic_cell.py` for prepared numerical image/lattice operations when required. The identification owns geometric face roles and domain binding, not a second lattice/wrap engine. Bind any `PeriodicCell.cell_id` and `AxisDomain.domain_id` through the same descriptor, with lazy dependency boundaries to avoid import cycles.
- Use existing factor/common-support preparation. Scalar interval seams use actual Dirac slices; vector-coordinate box seams use the real reduced tangential support. No dummy integration interval is introduced for a point seam.

**`phydrax/domain/_selection.py`, `_base.py`, `_components.py`, `_factor_component.py`, `_scalar.py`, `_hyperrectangle.py`, and `geometry1d/_primitives.py`**

- Add one native `CoordinateFace` selection for a declared coordinate component and lower/upper endpoint role. Bind its component identity through the owning coordinate port; an integer size/index alone is not scientific identity.
- Implement Cartesian face binding, exact face measure, point/grid sampling, and fixed-component bookkeeping in the existing factor/component owners. Scalar and one-dimensional geometry faces have unit Dirac/counting mass; box faces have their exact transverse Hausdorff measure.
- Preserve the original source domain and full coordinate event shape when embedding face samples. Reduced tangential sampling does not turn a vector coordinate into a different source schema.
- Extend `normal`/`normals` and face-gate preparation to this owned stratum, with explicit outward orientation and correct unit normal derivative scaling. Physical-boundary views use these actual components, not `where`-filtered whole boundaries with unknown mass.
- Validate explicit points against the selected face and refuse unsupported non-Cartesian geometries at the owning boundary. Geometry constructors and ordinary `Boundary()` behavior remain unchanged.
- Reuse the same face preparation for periodic seams and unpaired physical faces; no second face sampler or distance/gate convention is introduced in enforcement. Export the selection from its canonical public domain surface.

**`phydrax/domain/decomposition/_cover.py`**

- Add the identification binding to periodic `PairedSupport` construction and revision payloads.
- Allow equal patch IDs only for a verified nontrivial periodic identification; retain refusal for ordinary shared/overlap/transmission interfaces.
- Validate both coordinate maps against the corresponding source supports.
- Replace the skipped periodic map audit with the declared displacement/transport consistency audit. Keep sampled evidence sampled; structural construction evidence has an independently identified owner.
- Preserve deterministic pairing/source order and map/normal evidence.
- Exclude same-patch seams from ordinary adjacency/conflict edges; retain them in an explicit endpoint-incidence view. A seam is not a patch's conflicting neighbor.

**`phydrax/domain/decomposition/_cartesian.py`**

- Generate the wrap seam also for a one-cell partition.
- Extract only genuinely reused face/factor preparation into the domain-owned periodic/common-support preparation; delete the superseded local copies in the same cutover.
- Reuse the identification in wrapping, overlap images, cover preparation, and seam maps. Do not substitute raw modulo evaluation for smooth periodic field matching.
- Preserve existing patch windows, partition-of-unity normalization, ownership, sample addressing, and interior interface order.

**`phydrax/domain/decomposition/_geometry.py`, `_adapters.py`, `_routing.py`, and `_fields.py`**

- Carry identification/map evidence through cover audits and condition construction.
- Periodic-interface consumers create the new declaration; ordinary interface consumers keep their existing semantics.
- Keep self-seam incidence separate from ordinary adjacency, graph coloring, atlas edges, and overlap ownership. Preserve existing routing/workset order and maximum-overlap semantics.
- A partition-of-unity wrapped representative is not proof of matching one-sided raw local traces. Check/certify the actual endpoint jets; do not certify a discontinuous wrapped function because its two exact endpoint representatives happen to coincide.

**`phydrax/solver/functional_decomposition/_problem.py`, `_prepare.py`, `_solve.py`, `_schwarz.py`, and `_curvature.py`**

- Bind periodic self-seam execution/exchange incidences by explicit endpoint side, not patch ID alone. `PairScope` still owns one whole seam term; any endpoint-specific exchange records carry side separately.
- Apply the seam residual once with correct local-field ownership; do not duplicate an ordinary two-neighbor update or lose the right trace.
- Validate side-aware scopes during preparation and preserve checkpoint/exchange identities. Keep ordinary Schwarz/mortar execution and curvature ownership unchanged.

**`phydrax/solver/coupling/_interfaces.py`**

- Audit every assumption that a pair joins two distinct Euclidean patches.
- Make endpoint side explicit in `PairedSupportAttachment` for self-seams; missing/ambiguous side refuses before normal probes. Distinct-patch inference remains unchanged.
- Use the identification when a consumer genuinely supports periodic transport. Generic physical two-sided interface binding must explicitly refuse periodic/self-paired topology before Euclidean separation/normal-sign checks unless the full identification-aware branch is implemented and verified.
- Include identification content in revision-bound evidence where the pairing is supported.

**`phydrax/domain/__init__.py`, `domain/decomposition/__init__.py`**

- Export each public carrier from its canonical owner; no duplicate aliases or convenience re-exports of private helpers.

**Intentionally unchanged:** constructors of `Interval1d`, `HyperRectangle`, the abstract `Domain` hierarchy, and `ProductDomain` do not acquire independent periodic flags. Existing raw fundamental-domain boundary behavior stays unchanged. The explicit physical-boundary/axis adapter supplies the topological view.

### P2. Scientific condition and certified lowering

**New `phydrax/conditions/_periodic.py`**

- Implement `Periodic` and its native transported trace action.
- Keep distinct field sources ordered and same-field sources unique. Store side bindings separately from the unique source list.
- Evaluate derivatives through native request planning, then pull back each source jet to the common support; do not differentiate an already collapsed normal coordinate and accidentally obtain zero.
- Validate each evaluated event kind/dtype/rank/semantic layout against the declared fiber codomain at the owning action boundary, for both point and grid evaluation.
- Reuse `EventLinearMap`/typed flux capabilities where applicable; no raw callback receives a linear certificate.
- Validate target support and transport identity. Constant event targets are normalized exactly once; no accidental rank-based broadcasting across unrelated axes.

**`phydrax/conditions/_lowering.py`**

- Dispatch the new native declaration to certified lowering before generic legacy residual wrapping.
- Source field codomains describe their source domains, not the residual seam domain.
- Produce a `FieldCodomain` per trace and an explicit `Equality(target)`.
- Preserve existing uncertified behavior for arbitrary `Residual`, including when its numerical residual happens to be linear.

**`phydrax/conditions/_evidence.py`**

- Add the typed periodic trace/lift certificate with identification, requested trace, equality scope, derivative regularity, provider/right-inverse, preservation, tolerance, and numerical evidence.
- Reuse `ConditionRealizationStamp` and `ConditionEvidence`; do not fabricate finite function-space rank/nullity to fit `AffineProjectionCertificate`.
- Bind evidence to source and numeric revisions through the existing lifecycle contracts. No schema versions or compatibility generations.

**`phydrax/conditions/__init__.py` and `conditions/boundary.py`**

- Export `Periodic` once through the established public condition surface. Keep implementation in its owning module to avoid further enlarging unrelated boundary formulas.
- Update boundary documentation/import exposure only where the repository facade convention requires it; do not add an alternate constructor name.

**Intentionally unchanged:** `terms/_residual.py` does not gain a generic typed-product adapter. Existing subdomain jump classes retain their ordinary distinct-field interface meaning; shared pullback/target invariants may be reused without concrete inheritance or copied numerical logic.

### P3. Analytic fiber realization and polynomial preparation

**New `phydrax/_polynomial/_endpoint.py`**

- Own reusable bounded endpoint-jet polynomial tables and derivative/lift evaluation for analytic preparation.
- Prepare immutable host tables once, with explicit dtype and conditioning/size bounds; retain no hidden host conversion in query execution.
- Reuse existing polynomial evaluation primitives where sufficient. Bernoulli coefficients are generated only at preparation, not per model evaluation.
- For transported actions, expose the action matrix to native constraint/factorization preparation rather than hand-solving it.

**`phydrax/enforcement/_fiber.py`**

- Bind analytic-unit evidence and declared conditions to verified right-inverse preparation.
- Replace the analytic unit's single `residual_domain: Domain` with an explicit `residual_codomain: FieldCodomain | ProductCodomain`, using the existing typed codomain owner. A joint x/y seam action has different transverse supports; do not identify them because their dimensions match.
- Validate the product action/target leaf supports, event contracts, and canonical condition-to-leaf order before lifting. Corrections still return one update per written source field, preserving shared-write fusion.
- Clean-cutover all `AnalyticFiberProjectionUnit` constructors, fingerprints, public annotations, tests, and docs to this codomain contract. Do not retain a `residual_domain` alias or construct a fictitious common domain. Realized/separable fiber units keep their own batch/representation contracts.
- Support analytic derivative rules using endpoint requests and polynomial derivatives; reuse `FiberProjectionDerivativeRule` rather than a separate differentiation engine.
- Use a heterogeneous bounded product of residual fields for multi-seam preparation, not an unbounded stacking/vmap assumption. Static seam metadata stays outside numerical leaves.
- Maintain deterministic correction-field order and prohibit mixed analytic/realized evidence from masquerading as one continuum unit.

**New `phydrax/enforcement/_periodic.py`**

- Implement `prepare_periodic_projection` and `PreparedPeriodicProjection` as the domain-specific preparation/result integration, composed from existing fiber/linear substrates.
- Construct one joint typed condition from the canonical list of declarations; retain every constituent condition ID in evidence.
- Bind `ConditionEvaluationContext`, enforce exactness/regularity/resource admission, and adapt one fused analytic fiber state to `FieldRealizationResult`.
- Implement successful, unchanged, unsupported, nonfinite, validation/compatibility, solve/refresh failure, and transaction rollback paths. No candidate field is published after failure.
- Keep changing model parameters/state dynamic. Do not capture an obsolete model copy in a prepared lift, detach endpoint dependence, or refactor inside queries.
- Report endpoint-work and retained-state bounds. Batch/cache related endpoint requests, with explicit limits for multi-axis intersections.
- Use the existing lifecycle stamps, refresh/rebinding, and accepted-step RNG addressing. Deterministic fields use no random key; fixed random realizations share the same declared realization across paired evaluations. Refuse resampled/stateful sources when equality cannot be certified for the requested quantifier.

**`phydrax/enforcement/_realization.py` and `_lifecycle.py`**

- Add a common typed `RealizationAdmission` contract and an owning `AbstractFieldRealization.admission(...)` method describing actual read fields, possible write fields, established constraints, and bound preservation/representation certificates. Source fields in the condition are not a substitute for the write set.
- Require every native implementation to expose this contract. Unknown callable-chart updates conservatively cover every input field; they cannot masquerade as a disjoint realization because the condition only names another field.
- Bind admission certificates to preparation/source/geometry/representation revisions, not just display field names. Reuse existing result and transaction structures.
- Validate restored periodic prepared state at the existing reconstruction boundary; do not use raw Strict reconstruction outside it.

**`phydrax/enforcement/__init__.py`**

- Explicitly export the preparation function, prepared realization, and any genuinely public policy/evidence types.

**Intentionally unchanged:** no new local periodic kind in `EnforcementKind` or `TraceLifting`; no duplicate right-inverse/factorization implementation in enforcement. Nonlinear mathematics and feasibility algorithms remain unchanged, but field-realization composition admission is migrated as specified below.

### P4. Complete finite-coefficient realization

**`phydrax/discretization/spectral/_constraints.py`**

- Extend the owning endpoint action to expose paired lower/upper coefficient traces, coordinate-derivative scaling, phase/event transport, and tangential identities.
- Reuse the existing polynomial endpoint rows and provide a public owner-level action/prepared artifact rather than enforcement importing `_trace_row` privately.
- Preserve existing one-sided Dirichlet/Neumann/Robin/decay behavior and reduction order.
- Provide complete tensor-product seam coefficient rows, not collocation endpoint samples. Fourier periodic construction remains boundary-free.

**`phydrax/discretization/spectral/__init__.py` and relevant numerical-axis facade**

- Export the owner-level paired coefficient capability and bind identification metadata to `AxisDomain` without independently restating endpoints or periodicity.

**`phydrax/enforcement/_linear_representation.py` and `_representation_adapters.py`**

- Consume owner-provided periodic trace assemblies through `AbstractLinearRepresentation.assemble`/`LinearConditionAssembly`.
- Reuse `CoefficientElimination`, extraction/replacement/synthesis, real coordinate maps, and certificates.
- Validate the exactness of source synthesis, trace assembly, and target representation. A numerical fitted representation cannot gain continuum exactness through the new condition name.
- Expose the coefficient realization through the same preparation entry point with an explicitly supplied representation. Return/publish the actual constrained represented fields with their original identities and status.

**`phydrax/linalg/_constraint_operators.py` and `_constraints.py`**

- Reuse existing preparation/factorization first. Extend the owner only where an explicit independent-row reduction and original-system compatibility map are missing.
- Preserve rank/condition/nullity/failure evidence and strict/generalized semantics. Dependent but consistent rows need a certified equivalent independent system; incompatible targets fail before commit.
- Use native refresh and multi-RHS execution. No new dense inverse, per-column solve loop, or arbitrary tolerance-based row deletion.

**`phydrax/enforcement/_affine.py`**

- Integrate reduced-row/original-condition evidence only if required by the represented path.
- Keep existing finite, continuum-fiber, and generalized scopes honest; no fake finite Gram matrix for a continuum seam.

**Intentionally unchanged:** FE/IGA/ROM providers are not advertised as continuum-exact periodic owners unless they supply their complete trace assembly and representation certificates through the existing interface. Generic external representations remain usable when they supply that contract; this feature does not duplicate their basis preparation.

### P5. Preservation, compatible overlays, and finite observations

**`phydrax/enforcement/_api.py`, `_spec.py`, `_compile.py`**

- Admit the periodic typed realization through existing specification/compilation APIs.
- Build an affected-field/condition overlap graph at preparation. Fuse periodic writes; allow sequential overlapping realizations only with a bound preservation certificate or a joint provider.
- Reject raw boundary selections that conflict with paired seams; use the explicit physical-boundary view when compiling nonperiodic faces.
- Keep topological order for cross-field dependencies and existing atomic lifecycle semantics.
- Do not turn a successful finite compatibility probe into continuum evidence.
- Later global realizations on the same field must preserve the seam or be incorporated in the joint problem. Do not merely append periodic projection to the program.

**`phydrax/enforcement/_affine.py`, `_linear_representation.py`, and `_nonlinear.py` admission implementations**

- `PreparedAffineProjector` and `ExactAffineProjector` expose assembly input fields and actual correction fields; the wrapper retains the prepared artifact's admission.
- `CoefficientElimination` exposes its representation extraction dependencies and actual replaced/synthesized fields.
- `NonlinearFieldRetraction`, `LocalNonlinearRetraction`, and `MinimumDistanceRetraction` expose chart dependencies and writes. `AdditiveCorrectionChart` has an explicit field subset; `CallableCorrectionChart` conservatively reads/writes the entire provided field mapping unless a genuine typed chart contract proves a narrower set.
- All existing custom realization subclasses/callers migrate through refreshed references. Do not add a permissive compatibility default that labels unknown writes disjoint.
- For a periodic program, unknown/nonpreserving writes intersecting an established constraint refuse before execution. Known disjoint writes remain usable without unnecessary numerical checks. Verify that published field updates conform to the owning native admission contract before commit.
- Existing local overlays provide admission from their owned field/gate/target contracts. Do not require changes to unrelated nonlinear solve formulas.

**`phydrax/enforcement/_ansatz.py`, `_geometry_support.py`**

- Prepare exact coordinate-face gates/normal extensions for unpaired Cartesian faces using the existing gate owner. A gate on a y-face must not accidentally vanish on periodic x-faces.
- Use coefficient/construction proof that the gate and relevant target jets preserve periodic matching. Generic filtered MLS/BVH geometry that lacks this proof remains unsupported in an exact combined composition.
- Preserve existing exact-PDE trial-space protection.
- Apply exact-PDE trial-space preservation admission to typed global realizations as well as local ansatz transforms. The current local-only guard cannot be bypassed by a periodic realization carrying a valid condition but an unverified PDE-space-changing correction.

**Initial overlay paths in `enforcement/_compile.py` and `_ansatz.py`**

- Preserve periodic initial value and declared time jets using a time-only gate and a compatible periodic target construction/representation.
- Report proven inconsistent corner/time targets as incompatibility. Unknown arbitrary callable targets require an explicit certified realization or refusal of the exact combined route, not silent modification of the supplied data.
- Preserve evolution-variable semantics when time itself is the periodic coordinate: periodic time is not automatically an initial-value problem.

**`phydrax/enforcement/_cardinal.py`, `_kernel.py`, `_observation.py`**

- For finite observations/anchors on periodic fields, prepare correction sections in the homogeneous periodic space and, where required, homogeneous boundary/initial spaces.
- Assemble the observation action on those sections and prepare its native right inverse once. Recompute the cardinal system after changing the correction space; wrapping an old basis does not preserve its cardinal values.
- Normalize image observation actions and targets using the enforced field relation, including scalar/event transport, affine lift offsets, requested derivative jet, and orientation. Retain an original-row reconstruction/provenance map before deduplication. Only then merge compatible dependent observations; conflicting normalized targets refuse.
- Geometric identification alone does not imply equal raw values. For `u(upper)=gamma*u(lower)+g`, observations are compatible when their targets satisfy that equation; antiperiodic targets `1` and `-1` can be compatible while equal nonzero targets are not.
- A derivative-only seam relation does not identify endpoint values. Do not merge value observations unless that value relation is actually constrained or structurally certified. Observations outside the declared fundamental support require an explicitly certified extension; do not invent an affine/Bloch image extension from geometry alone.
- Keep discrete observation exactness distinct from continuum seam preservation.
- Use compact prepared neighborhoods or explicit bounded dense/reference sections; do not allocate unbounded query-by-anchor or corner products.
- Legacy Euclidean interior overlays on the same periodic field either lower to this joint prepared correction or are refused with the supported replacement explained.

**`phydrax/kernels/_transforms.py` and `_field_metric.py`**

- Reuse `InputTransformedKernel` for smooth periodic feature pullback and the existing finite correction/metric machinery.
- Bind the transform to identification and actual derivative regularity evidence. An arbitrary callable with a claimed derivative order is not automatically a periodic certificate.
- Do not call nearest-image/geodesic distance globally smooth; its branch cuts need their own regularity treatment.
- Matrix-free tolerance-terminated or selected-section approximations remain non-exact and cannot satisfy an exact continuum request.

### P6. Explicit periodic model construction and evidence propagation

**`phydrax/nn/layers/_fourier_embeddings.py`**

- Require genuinely integral positive harmonic mode values, a finite positive period, unique modes, and no passthrough of a certified periodic coordinate. Reject invalid values before assignment.
- Preserve the existing direct explicit-wavevector API; validate integer-lattice membership when certifying a declared identification, without rounding or snapping trained frequencies.
- Multiple coordinates bind explicit integer wavevectors to each identification. Include a separating fundamental harmonic when the construction claims expressivity on the entire quotient; report an intentionally smaller represented period rather than silently treating it as unrestricted.
- Keep frequencies/phases fixed for certified periodic constructions. Do not stamp unrestricted trainable/random frequencies periodic.

**`phydrax/nn/_contracts.py`, `_model/_protocols.py`, `_differentiation.py`**

- Add a periodic construction certificate through the existing `AbstractConstructionCertificate`/model metadata channel.
- Bind the claim to input ports, coordinate packing, identification, and the complete model dataflow. An embedding certificate alone certifies features, not arbitrary raw-input skip paths in a composite model.
- Downstream regularity bounds the derivative claim; preserve existing randomness/execution authority distinctions.

**`phydrax/nn/models/wrappers/_sequential.py`, plus `phydrax/domain/_model_function.py` / `_domain.py` binding paths**

- Certify explicitly constructed sequential feature-to-model compositions when all periodic-coordinate paths pass through the certified features and the model is pointwise under the declared binding.
- Do not introspect arbitrary attributes or rewrite the input shape of an already-built model. Operator/grid models need their own explicit execution/representation certificate, not a pointwise feature assumption.
- Map model input coordinate positions to domain semantic ports; wrong coordinate packing refuses certification even if dimensions happen to match.

**`phydrax/domain/_function.py`, `phydrax/operators/_composition.py`, and `phydrax/operators/differential/_domain_ops.py`**

- Register the new certificate in the existing drop-on-nonpreservation mechanism.
- Preserve compatible sums using the same scalar transport/zero-jump contract. Products of scalar Bloch fields have the product transport, not automatically the original phase; affine-jump arithmetic needs its actual transformed target. Certify only implemented algebraic rules.
- Coordinate differentiation preserves ordinary/constant-phase homogeneous periodicity up to the available regularity. Differentiated affine targets transform explicitly; no claim is copied unchanged.
- Arbitrary pullbacks drop the certificate unless the substitution is proven identification-equivariant. Derivative metadata copying must not carry a stronger regularity/order or different coordinate identity.
- Preserve existing convexity/trial-space certificate behavior and operation order.

**Prepared periodic realization in `enforcement/_periodic.py`**

- Recognize a valid matching structural certificate as an unchanged realization only when every requested relation, target, regularity, source binding, and preservation requirement is covered.
- Nonzero affine jumps and Bloch factors require the corresponding explicit lift/phase construction or the analytic/represented correction; plain periodic features alone do not certify them.

### P7. Public docs, examples, qualification, and generated data

**New `examples/enforced_periodic_constraints.py`**

- Give complete executable examples for an unmodified smooth pointwise model with analytic value/first-derivative enforcement and for explicit periodic feature construction.
- Include a periodic manufactured PDE/remaining objective, not endpoint equality alone. Show one compatible initial/boundary/anchor scenario, one transported/affine relation, status/certificate inspection, and a clean failure example.
- Keep bounded deterministic data and stable compiled identities; no optional provider is needed.

**`docs/guides_conditions.md`, `docs/api/conditions/boundary.md`, `docs/api/conditions/core.md`**

- Document the single scientific declaration, source/support identity, soft versus hard use, same-source seam binding, per-trace order semantics, event/flux orientation, and exactness table.

**`docs/api/enforcement.md`, `docs/api/solver/enforcement.md`**

- Document prepare-once joint realization, existing `EnforcementSpec` use, refresh/rollback, certificate inspection, preservation/fusion requirements, and unsupported exact compositions.

**`docs/guides_functional_domain_decomposition.md`, `docs/guides_spectral_methods.md`, `docs/api/discretization/spectral.md`, `docs/api/nn/embeddings.md`, `docs/all-of-phydrax.md`**

- Explain one-patch seams, physical boundary versus paired supports, explicit `AxisDomain` binding, complete coefficient traces, Fourier construction/regularity, and finite-sample nonclaims.
- Update any affected examples/callers found by refreshed symbol references; remove stale duplicate constructions introduced by the cutover, not unrelated legitimate interface APIs.
- Update `docs/guides_domain.md`, `docs/api/domain/decomposition.md`, `docs/guides_exact_trial_spaces.md`, and `docs/api/equations/trefftz.md` for explicit physical-boundary views, side-aware self-seams, and the separate periodic versus exact-PDE certificate/admission contract.

**`phydrax/qualification/_core_portfolio.py` and its existing catalog integration**

- Register only bounded implemented owner support tuples for analytic Cartesian, represented coefficient, and explicitly constructed periodic routes, with precision, orders, axis/query/observation capacity, derivative, and lifecycle limits.
- Mark them unreleased candidate/research evidence as appropriate; no broad production or arbitrary-geometry claim follows from unit tests.
- Qualification evidence names executed scenarios and exactness boundaries. Do not fabricate signed attestations or pass records.

**`docs/data/public_api.json`, `docs/data/capabilities.json`, `docs/api/capabilities.md`, catalog-derived closure/portfolio data, and `CHANGELOG.md`**

- Regenerate through the existing tools after public declarations/qualification changes. Do not hand-edit generated records or add representation/schema generations.
- Add a concise changelog entry distinguishing analytic continuum matching, represented-space enforcement, structural construction, and finite observations.
- Existing relevant navigation can be reused; modify `mkdocs.yml` only if an actual new navigable page is needed.

## 6. Permanent test plan

Tests own consumer-visible contracts, not exports/source spelling/wiring. Each independently meaningful refusal, lifecycle phase, or scientific scenario remains separately collectable. Use parametrization with diagnostic IDs for real case matrices. Shared helpers belong in `tests/_support` only when they own an independent invariant. No repository-wide fixture/configuration change is planned.

### New `tests/unit/domain/test_periodic_identification.py`

- Interval, scalar-coordinate interval, vector box, and product-domain seams retain the intended coordinate/source identities, transverse layouts, and measures.
- Single-patch periodic cover produces a usable self-seam; ordinary same-patch interfaces still refuse.
- Zero-dimensional point seams use correct Dirac/counting semantics, not random auxiliary coordinates.
- Physical boundary excludes exactly the identified faces; mixed periodic/bounded boxes retain the correct normals; a fully periodic support has an empty physical-boundary collection.
- Wrong displacement, source map, orientation, period, or axis binding is diagnosed; periodic auditing cannot verify an arbitrary incorrect mapping merely by skipping Euclidean mismatch.
- Equivalent image anchors share geometric identity; equal shapes on distinct coordinates do not.
- Self-seams do not create self-neighbor conflict edges; routing, adjacency, atlas adapters, and endpoint-incidence consumers retain deterministic ownership.

### New `tests/unit/constraints/test_periodic_conditions.py`

- Same-field value and derivative residuals evaluate the two physical traces correctly; a transverse/nonconstant manufactured field catches wrong-side pullback and differentiation-after-collapse errors.
- Scalar, vector-event, complex Bloch, antiperiodic, affine jump, and oriented linear-flux contracts use explicit semantics and dtype/layout validation.
- Specialized lowering preserves source domains, target separation, certified linearity, and one-source binding. Arbitrary legacy residuals remain uncertified.
- Asymmetric distinct-source transport exercises canonical `(source, target)` tuple/action order through soft residual, certified lowering, and hard preparation; same-field tests cannot detect swapped source roles.
- Soft loss is the independently computed Hermitian residual score on a supplied common-support integration realization, including complex and derivative cases.
- Each target/event/support/transport refusal is its own scenario.

### New `tests/unit/enforcement/test_periodic_projection.py`

- Analytic value-only and value/first-derivative projections agree with independent low-order formulas at interior and endpoint coordinates, not just with their own residual evaluator.
- Declared higher/sparse jet requests use independent polynomial endpoint identities. Undeclared higher jets are not claimed satisfied.
- Projection preserves already-satisfied fields and is idempotent for fixed affine targets; free interior variation remains available.
- Coordinate JVP/VJP and model-parameter gradients agree with independent finite-difference/analytic references. Endpoint model dependence must not be detached.
- Two/three-axis manufactured fields exercise corner compatibility and transverse mixed derivatives. Conflicting affine jumps fail rather than be averaged.
- Transport phases near identity/antiperiodicity retain rank and conditioning evidence without division singularities or silent regularization.
- Compatible physical-face, initial/time-jet, and finite observation scenarios preserve every intermediate/final contract; adversarial ordering or an unverified later realization is refused.
- Conflicting image observations, unavailable preservation, insufficient regularity, unsupported geometry, nonfinite data, and exceeded work/memory/order caps have separate observable failure assertions.
- Compatible and incompatible antiperiodic/Bloch/affine image observations are separately collectable. A derivative-only relation keeps independent endpoint value observations distinct.
- A later realization whose condition reads one field but whose chart writes another periodically constrained field is rejected before any candidate commit unless genuine preservation evidence is supplied.

### New `tests/unit/enforcement/test_periodic_linear_representation.py`

- Complete coefficient constraints give equality at off-grid transverse coordinates and derivative orders; a deliberately aliased collocation example cannot receive continuum evidence.
- Complex realification and coefficient ordering survive extraction/replacement/synthesis.
- Redundant compatible rows preserve every declared condition; inconsistent duplicates, deficient correction spaces, and unsupported generalized exact requests fail with native evidence.
- A certified ordinary Fourier representation skips rows only for its covered homogeneous identity-transport relation. Separate antiperiodic/Bloch represented cases assemble their actual coefficient actions; a nonzero affine jump on an unlifted ordinary Fourier representation refuses, while a declared compatible affine lift remains admissible.
- Exact-PDE trial-space protection remains intact unless its owner certifies the constrained representation.

### Existing affected tests

- `tests/unit/domain/test_subdomain_advanced.py`, `test_subdomain_decomposition.py`: preserve wrap-image/overlap/partition-of-unity behavior and add the single-partition seam regression.
- `tests/unit/constraints/test_subdomain_interfaces.py`: retain ordinary interface semantics and exercise the canonical periodic cutover where applicable.
- `tests/unit/enforcement/test_realization_lifecycle.py`: atomic failure, failed refresh, accepted-step/source binding, checkpoint/restore, and no leaked candidate fields.
- `tests/unit/enforcement/test_affine_projector_closure.py`, `test_linear_representation_closure.py`: only add/update cases affected by actual native row-reduction integration.
- `tests/unit/discretization/test_spectral_methods.py`, `test_unbounded_spectral.py`, `test_spectral_field_view.py`: retain one-sided endpoint contracts and add paired coefficient behavior at the spectral owner.
- `tests/unit/nn/test_embeddings/test_random_fourier.py`: retain numerical feature behavior; add passthrough, nonintegral/invalid-period, multi-axis integer-lattice, and unintended subperiod contracts.
- `tests/unit/domain/test_function_transforms.py`, `test_coordinate_ports.py`, `tests/unit/test_component_contracts.py`: truthful certificate binding, preservation/drop behavior, nonsmooth downstream regularity, and semantic input-port mismatch.
- `tests/unit/kernels/test_transforms.py` and `tests/unit/constraints/test_cardinal_observation_providers.py`: periodic homogeneous correction spaces, regularity, actual cardinal values, and declared resource bounds.
- `tests/unit/solver/test_functional_decomposition.py`, `test_functional_decomposition_advanced.py`, and `tests/unit/meshing/test_interface_bindings.py`: explicit self-seam endpoint roles, no doubled neighbor updates, supported identification-aware transport or explicit refusal before ordinary Euclidean orientation checks.
- `tests/unit/test_construction_certificates.py`, exact-trial-space/Trefftz tests, and `tests/unit/discretization/test_axis_domains.py`: certificate identity and separate periodic/PDE preservation, plus matching identification/numerical-axis identities.
- `tests/integration/enforced_pipeline/`: periodic plus compatible boundary/initial/observations through the actual program; conflicting targets and subsequent nonpreserving global realizations must not yield success.

### New `tests/integration/test_solver_periodic_enforcement.py`

Run a bounded actual `FunctionalSolver` lifecycle with a manufactured smooth periodic field/PDE. Inspect ansatz fields before and after accepted updates, enforce requested endpoint jets and compatible data, and preserve failure/rollback evidence. Do not use optimization convergence as the sole proof of exactness or impose flaky timing/convergence assertions. A separate complex/transport case proves consumer-visible complex semantics.

### New `tests/typing/cases/periodic_constraints.py`

Pin actual public constructor/result/certificate types with `assert_type`, static coordinate/event/target contracts, and deliberate misuse paired with runtime refusal where reachable. Use the pinned ty conventions and narrow exact-rule ignores only for negative cases. No `Any`, cast, or suppression merely to paper over the public contract.

### Oracle and fault adequacy

Use exact host rational/elementary low-order references and independent basis evaluation. Inject deliberate wrong endpoint sign, missing derivative scaling, detached endpoint parameters, aliased transverse samples, false certificate inheritance, and conflicting image targets into throwaway fault checks. The relevant permanent tests must fail for these consumer-visible errors. Remove the fault scaffolding after verification.

Mark dtype/rank-promotion/complex-contract tests `strict_jax`. Keep provider-free tests independent of optional dependencies. Do not add source-text tests, mock-forwarding tests, nonempty-result tests, broad scenario wrappers, or tests asserting incidental implementation defaults.

## 7. Benchmark and runtime proof

**New `benchmarks/periodic_enforcement.py`**, using `benchmarks/_runtime.py`:

- Separate immutable host preparation, lowering, compilation, first synchronized execution, warmed prepared evaluation, derivative evaluation, and explicit refresh.
- Record compiler temporary/output/code bytes, logical retained bytes, dtype/backend/environment identity, and declared capacity/work bounds.
- Vary query count, model width, jet order, periodic axis count, represented mode count, and observation capacity in bounded campaigns. Do not use a single small example as scaling evidence.
- Compare the actual free-model, analytic lift, coefficient, and explicit feature constructions at their stated contracts. Feature construction changes trial-space expressivity; do not claim an apples-to-apples training or scientific accuracy advantage from execution timing alone.
- Include prepared-versus-cold/refresh evidence and the cost of coordinate/model-parameter derivatives. Correctness/status checks precede timing.
- Exercise bound refusal and a multi-axis endpoint-work campaign. A query-by-anchor or recursive corner expansion must stay within declared limits; benchmarks do not excuse unbounded execution.
- Store only real measured records in the repository's current canonical record format. No schema/version suffix, fake fallback, placeholder timing, or new benchmark framework.

Runtime smoke during implementation must execute the new example and the real solver scenario, observe endpoint/flux/target certificates and interior PDE residuals, and inspect one failure/rollback path. Tests alone are not runtime proof.

## 8. Minimal conservative verification selection

Before execution, select tests from all changes since the last merge into `dev`, excluding realistically unaffected surfaces. Refresh exact affected node IDs with the repository selection tooling/configuration; do not run a guessed enormous suite. Use `-n auto` unless a device/provider isolation requirement dictates a dedicated invocation.

Planned initial selection consists of the new periodic tests plus the affected existing files listed in Section 6, narrowed to touched contracts when safe. No global fixture/collection change is planned; run the full suite only if implementation introduces a truly cross-cutting change that requires it.

Implementation commands to use after code is complete, with the actual selected tests:

- `python tools/check_typing.py check` (all first-party Python, zero diagnostics).
- `python tools/audit_selectors.py` (zero findings).
- Configured Ruff lint/formatter on changed code, not a new style/checker.
- `python -m tools.check_installed_typing` for the changed public annotations.
- `python -m tools.generate_public_api_manifest` and `python -m tools.check_public_api_manifest`.
- `python -m tools.generate_capability_inventory` and `python -m tools.check_capability_consistency` after bounded capability declarations are updated.
- `python -m pytest -n auto <selected periodic and affected contract tests>`.
- `JAX_ENABLE_X64=true python -m benchmarks.periodic_enforcement` with the actual bounded campaign arguments implemented by that benchmark.
- `JAX_ENABLE_X64=true python examples/enforced_periodic_constraints.py` and the bounded real solver smoke.

Measure complexity for materially changed symbols before/after with the existing repository tools. Preserve numerical operation/reduction ordering in unchanged owner kernels. Audit remaining host synchronizations, endpoint loops, dense row products, repeated factors, and full sorts; each remaining hit needs a static/bounded/host-preparation/output-size rationale.

## 9. Dependency order and integration ownership

1. Freeze declaration/source/trace/identification identities and refresh public references.
2. Implement P1 geometric identification and P2 declaration/certified lowering independently once their common map contract is frozen; one integration owner controls shared facade/metadata edits.
3. Implement P3 analytic preparation and P4 owner coefficient assembly/native row handling as independent numerical slices, reusing the frozen declaration.
4. Implement P5 preservation/joint composition against both prepared routes. Do not ship a bare last-stage projector as completion.
5. Implement P6 structural construction/evidence and bind it to the same identification and preservation contract.
6. Complete P7 consumer docs/examples/capability data, then run the selected checks, actual runtime smoke, and phase-separated benchmark once integration is complete. Agents must not run build/lint/tests/formatters mid-flight.

Tests for each invariant are developed alongside its owner; execution is integrated afterward. Re-read any file affected by concurrent changes or a failed edit before applying changes. Remove superseded helpers/callers in the same clean cutover; preserve unrelated existing APIs that own different contracts.

## 10. Completion checklist

- [ ] One periodic scientific declaration is usable with existing soft residual terms and exact typed enforcement without changing its meaning.
- [ ] Same-field sources bind once; same-patch seams are real identified supports, not fake aliases/patches.
- [ ] Requested values/jets/linear fluxes, scalar antiperiodic/Bloch transport, and affine targets have explicit conventions and independent numerical evidence.
- [ ] Multiple periodic directions have bounded work and verified corner/transport compatibility.
- [ ] Analytic, complete represented-space, and explicit structural routes publish their actual scope/regularity/status; sampled constraints cannot claim continuum equality.
- [ ] Physical boundary and numerical-axis adapters consume the same identification; raw fundamental-domain behavior is intentionally unchanged.
- [ ] Compatible boundary/initial/finite data remain satisfied; unsupported or inconsistent compositions fail before commit.
- [ ] Finite coefficient redundancy/rank/conditioning/target compatibility evidence survives to the consumer.
- [ ] Periodic construction certificates are bound to complete input dataflow, truthful downstream regularity, semantic ports, and supported randomness.
- [ ] Model/coordinate derivatives, refresh/checkpoint/rollback, nonfinite/error semantics, and resource refusal are exercised.
- [ ] Every affected callsite/test/doc/example/manifest is migrated or explicitly unchanged; obsolete duplicate helpers are removed.
- [ ] Pinned typing/selector checks, minimal affected tests, installed public typing, real runtime smoke, and relevant phase-separated benchmarks have actually run and their results are recorded without fabricated claims.

## 11. Planning review and verification boundary

Two independent read-only reviews covered numerical/exactness/preservation contracts and public-interface/ownership/migration contracts. Their material findings were incorporated:

- Fourier row omission is conditional on the complete requested transport and target, not merely the basis name.
- Image observations normalize the actual value/jet action, phase, affine offset, and target; derivative-only matching does not imply equal endpoint values.
- All field realizations expose actual possible writes and bound preservation admission; condition inputs cannot hide cross-field updates.
- Distinct-source field/action tuples follow canonical source/target roles independently of cover left/right traversal.
- The centered Bernoulli interior oracle matches the chosen lift gauge.

Both reviewers reported no remaining material concern in their assigned plan review after these corrections. This is review evidence for the plan, not proof that the proposed implementation works. The explicit native face-selection/component ownership above closes the inspected gap that `AbstractGeometry.bind_component` currently accepts only interior/full-boundary selections and rejects geometry `Fixed` sampling. All runtime, typing, test, benchmark, and generated-data commands in this document remain future implementation verification, not completed checks.

This plan does not claim automatic periodicity for arbitrary geometric identifications, arbitrary callbacks, arbitrary external FE/IGA bases, or unbounded operator-network execution. Those inputs must supply the owning certified trace/representation/preservation capability or receive an explicit unsupported result; no advertised route substitutes a sampled or approximate fallback.
