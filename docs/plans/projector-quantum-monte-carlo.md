# Native projector quantum Monte Carlo: end-to-end implementation plan

## 0. Status, scope, and execution rules

This is an implementation plan, not an implementation or a scientific qualification record. Repository sources, tests, documentation, and generation tools were inspected without running package code, tests, imports, benchmarks, or formatters. Language-server references were queried for the shared exported key-group contract and its scalar-bound consumer. Existing paths below were inspected or discovered; paths marked **NEW** are proposed additions. Line numbers are not execution contracts.

The upstream algorithmic reference is Rimu.jl, inspected at commit `869e334c101c82ca8640e54338c6ffa170886ec8`, together with [the software paper](https://arxiv.org/html/2601.19505). Implement the mathematics through Phydrax-native owners; do not transplant its dictionary runtime, wrappers, statistical claims, or source-text tests.

The complete deliverable in this plan is:

1. Unranked, identity-carrying, packed quantum configuration addresses and genuine outgoing Hamiltonian columns.
2. Prepared sparse local transitions, exact column application, and probability-corrected raw-route sampling.
3. A bounded single-device signed/complex ground-state projector solver with adaptive semistochastic spawning, complete annihilation, optional late unbiased compression, and independent replica controllers.
4. Joint correlated ratio estimation, physical projected and replica observables, shift diagnostics, and finite-history population-control reweighting.
5. Frozen positive guiding similarities with the matching physical metric and observable recovery.
6. Transactional checkpoint/continuation, explicit resource-only replay, independent scientific controls, phase-separated benchmarks, documentation, and generated public/capability data.

All six items are required before implementation is called complete. Dependency phases below are not partial deliveries or independent feature claims.

The preceding assessment explicitly deferred initiator approximations, noisy excited-state orthogonalization, transcorrelated/model catalogs, external Julia integration, and distributed execution. They remain outside this implementation, not unfinished branches. Do not expose selector values, placeholder classes, automatic fallbacks, or capability claims for them. General nonsymmetric Hamiltonians are not admitted merely because outgoing columns can represent them. A guided internal operator may be non-Hermitian only with a certified self-adjoint original operator and an explicit matching guide/metric contract.

For a later `//code` instruction: create a fresh worktree from the then-current branch under the repository-prescribed worktree directory, copy the authoritative ignored `AGENTS.md` before any implementation, and keep all implementation there. `//code--` is the only exception. Do not create a worktree or change implementation code for this `//plan` request. Do not assume the inspected working tree is clean or reset unrelated changes.

Expected performance and scientific usefulness are hypotheses until the benchmark/qualification gates below are exercised. Operational completion is not a release or a ground-state-convergence certificate.

## 1. Baseline facts and immutable boundaries

| Existing owner | Verified contract | Binding decision |
| --- | --- | --- |
| `operators/quantum/lattice/_model.py` | Local spaces, ordered factors, exact charge deltas, global `FermionModeOrder`, and construction self-adjointness | Remains the single physical term algebra. |
| `lattice/_compile.py` | Existing preparation admits dense product-of-local-dimension branch counts | Keep its input admission, prepared objects, IDs, leaf order, and error behavior; add a separately admitted sparse-column preparation target. |
| `lattice/_operator.py` | Per-coordinate action emits raw `H[out, in]` branches; finite sector action scans every admitted source index | Reuse its factor order/parity mathematics; do not turn full-sector arrays into sparse coefficient populations. |
| `lattice/_sector.py` | Direct DP rank/unrank, signed-int64 counts, dimensions below signed-int32 route limits | Keep as the finite exact-vector route. The new address domain has no rank or dimension-admission requirement. |
| `lattice/_vmc.py`, `quantum/_discrete.py` | VMC consumes `H[current, connected]`; the lattice adapter conjugates outgoing values only after self-adjoint certification | Preserve this row meaning, zero-current refusal, shapes, and callers. |
| `phydrax.linalg.eigen` | Finite self-adjoint and nonsymmetric matrix-free eigenproblems already exist | No projector-specific deterministic eigensolver or dense fallback. |
| `sparse/_key_groups.py` | Scalar integer keys, compact grouping/lookup/alignment, stable IDs, independent group/member overflow evidence | Generalize this one implementation to exact word-vector keys; do not add a parallel multiword grouping family. |
| `sparse/_execution.py` | Fast, fixed-order, and `two_sum`-compensated signed/complex reductions | Extend this owner with seeded compact-key reduction; preserve existing relation outputs and operation order. |
| `_sampling/_addressing.py` | `SampleAddress` and exact uint32-word `derive_key` folding | Reuse unchanged. No global RNG, storage-slot addressing, or capacity-dependent physical draws. |
| `uq/_correlated_observable.py` | Real scalar FFT/Geyer initial-monotone diagnostics; zero variance is an existing refusal | Preserve this API and behavior. Joint ratios are not two scalar diagnostic calls. |
| `uq/_free_energy.py` | Grouped initial-positive correlation selection, synchronous block metadata, influence factors, and block bootstrap | Extract only genuinely shared selection/grouping invariants, preserving its scientific records and existing statistical convention. |
| `solver/_runtime_lifecycle.py`, `phydrax.lifecycle` | Array-only checkpoint envelopes, restart relations, commit/rollback, durable result/artifact storage | Reuse; add projector adapters, not another archive format or repository. |

Existing VMC, TDVP, TPQ, response, quantum jumps, orbit/irrep sectors, MPO/Abelian-MPO targets, continuum variable-sector VMC, posterior/ensemble reweighting, and chemistry providers are not migrated to the projector. They retain their current algorithms, RNG, evidence, and persistence semantics. New composition is not a reason to change them.

## 2. Ownership, public surface, and dependencies

### 2.1 Canonical ownership

| Capability | Owner |
| --- | --- |
| Configuration identity, exact packed codec, unranked charge predicates | `phydrax.operators.quantum.lattice` |
| Sparse local transition preparation and outgoing exact/raw-route action | Same lattice owner, sharing existing term/factor mathematics |
| Guide snapshot, diagonal similarity, inverse physical metric | Same lattice owner; no solver or UQ imports |
| Exact multiword equality/grouping, seeded signed accumulation | `phydrax.sparse` |
| Projector equation, spawning policy, compression, population controller, state and history | `phydrax.solver` |
| Physical sparse overlaps and projector-specific estimator production | Projector solver owner |
| Generic aligned correlated ratio-of-means and statistical denominator gates | `phydrax.uq` |
| Semantic RNG | Existing `_sampling` owner |
| Durable checkpoints/results and resource-shape transports | Existing lifecycle/runtime owners, through projector adapters |
| Scientific qualification and benchmark evidence | Existing qualification and benchmark utilities |

Import direction: lattice address/column/guide modules may depend on sparse and dependency-minimal linalg evidence, but never solver or UQ. UQ ratio/selection modules never import projector solver types. Projector contracts import operator types; projector estimator adapters import UQ; lifecycle adapters import the existing checkpoint/result owners. No UQ-to-solver cycle. Do not re-export private imports through unrelated facades.

### 2.2 Proposed public names

Names below are the planned canonical API, not aliases for existing VMC or finite eigensolver names:

- Lattice: `QuantumAddressCodec`, `QuantumConfigurationDomain`, `QuantumAddress`, `QuantumColumnResourcePolicy`, `PreparedQuantumLatticeColumns`, `prepare_quantum_lattice_columns`, `refresh_quantum_lattice_columns`, `QuantumLatticeColumnOperator`, `QuantumColumn`, `RawQuantumExcitation`, `QuantumGuide`, `GuidedQuantumColumnOperator`.
- Sparse: extend `KeyGroupPlan`/state/lookup/alignment; add `KeyGroupAccumulation`, `KeyGroupReductionEvidence`, and `reduce_key_groups`.
- UQ: `CorrelatedRatioPolicy`, `CorrelatedRatioResult`, `CorrelatedRatioStatus`, and `correlated_ratio_of_means`.
- Solver: `ProjectorMonteCarloProblem`, `ProjectorMonteCarloPlan`, `PreparedProjectorMonteCarlo`, `ProjectorMonteCarloState`, `ProjectorMonteCarloStepResult`, `ProjectorMonteCarloResult`, `ProjectorMonteCarloStatus`, `ProjectorEstimatorPolicy`, `prepare_projector_monte_carlo`, `initialize_projector_monte_carlo`, `step_projector_monte_carlo`, `solve_projector_monte_carlo`, and projector checkpoint/restore adapters.

Use one solve entry point: initialization is explicit; the same solve function accepts an existing validated state to continue. Do not add a forwarding `continue_*` alias, an FCIQMC alias, a quantum eigensolver wrapper, or another top-level `phydrax.operators` spelling.

All new owned modules inherit `StrictModule`; concrete classes are final by default. New array/metadata contracts use runtime-resolvable `phydrax.typing` forms, nominal `*Dim` classes, one `Scope` at conversion boundaries, and `__strict_contract__ = True`. Use explicit complex128/float64/int32/uint32/int64 contracts and typed `PRNGKey`. The planned numerical execution precision is single-device complex128 with float64 real diagnostics; it is not an implicit fallback to single precision. Refuse incompatible precision at admission. No typing/axes-core extension is required.

Selectors are declared once at runtime and parsed by the native owner. Keep spawning (`exact`, `sampled`, `semistochastic`), compression (`none`, `threshold`), controller (`fixed-shift`, `double-log`), and guide-zero policy (`reject`, `positive-log-floor`) exhaustive and separate. Do not add unsupported selector members. Reuse `RelationAccumulation`; the projector admits deterministic and compensated accumulation, not an unqualified fast scatter path.

### 2.3 Dependency phases

- **P0:** freeze contracts, references, oracles, identities, and resource formulas.
- **P1:** packed addresses and shared multiword/seeded sparse execution, independently implementable after P0.
- **P2:** sparse transition preparation and exact/raw-route column action; depends on address and sparse contracts.
- **P3:** generic joint correlated ratios; independent of the projector after shared statistical contracts are fixed.
- **P4:** projector propagation, physical observations, and raw history; depends on P1/P2.
- **P5:** guiding and projector estimators; depends on P2/P3/P4 interfaces.
- **P6:** lifecycle integration, public exports, qualification, benchmarks, docs, and generated records.

One integration owner controls shared facades and final verification. Independent slices may be implemented concurrently only after interfaces are fixed. No mid-flight build/lint/test/formatter runs by subagents; run the selected integrated checks once afterward, then address failures specifically.

## 3. Configuration/address and column contracts

### 3.1 Scientific domain identity without rankability

`QuantumConfigurationDomain` derives site IDs/order, local dimensions/space IDs/statistics, labels, charge vectors, and global fermionic order from `QuantumLatticeSpecification`. Require an explicit per-site species/component assignment where that scientific identity is needed; do not infer species from names or equal shapes. This assignment augments identity, not local operator algebra.

The domain owns an optional exact charge predicate using the existing `AbelianGroup` semantics for integral/modular charges. It builds no DP table or product charge grid. A Hamiltonian must be an endomorphism of the admitted domain: certify every compiled monomial charge delta against the constraint or refuse before execution. Do not filter away charge-changing transitions to make an invalid Hamiltonian appear valid. Unconstrained product domains can admit non-number-conserving terms while still using finite declared local occupation cutoffs.

Before runtime charge reductions, bound every integral component's attainable sum using host integers and admit an exact integer accumulator width; refuse overflow rather than summing many int32 local charges in int32. Modular predicates use the canonical group reduction with bounded intermediate values. Preserve charge identity separately from narrowed numerical metadata.

Keep physical `domain_id` separate from numeric binding and resource IDs. Changing coefficients changes prepared/operator identity, not configuration equality; changing site order, statistics, species, mode order, dimensions, or charge constraints changes domain identity. Keep `codec_id` explicit. Domain equality is never inferred from key width or dimensions.

### 3.2 One packed representation

For local dimension `d`, allocate `(d - 1).bit_length()` bits, including zero bits for a singleton space. Store at least one uint32 word even for a singleton/vacuum domain. Pack site fields in canonical specification order; store and compare words most-significant first. Declare unused bits zero. Implement fields crossing a word boundary with prepared shifts/masks; never rely on a shift by the word width or narrowing before validation.

- Public coordinates are validated integer local-state indices before int32 conversion.
- Address storage is `UInt32[AddressWordDim]`, or `UInt32[SupportDim, AddressWordDim]` for support arrays.
- Decode to transient `Int32[SiteDim]` only for an action; do not retain both coordinates and packed keys in every state/event.
- Reject out-of-range local states, noncanonical unused bits, unknown domain/codec, unrepresentable local dimensions, and address/storage budgets before narrowing or allocation.
- All-zero words are a legitimate active configuration. A separate active mask identifies padding; no physical key is reserved as a sentinel.
- A 100-binary-mode address requires four words without computing a global Hilbert rank. This does not imply unbounded support or event capacity.

The codec is not a mixed-radix scalar rank, hash, Morton geometry address, or new species/statistics algebra. Existing finite fermion-basis bit conventions remain unchanged; any conversion is explicit through coordinates and mode identity.

### 3.3 Sparse immutable transition preparation

A new sparse-column target prepares per-local-input lists of permitted output states from the canonical `LocalOperatorPlan.support_mask`, with fixed padded widths and bound complex values. `LocalOperatorPlan.operator_id` is structural: it fingerprints space, label, charge delta, parity, support, and shape, but not matrix values. Deduplicate output-index/mask topology by that structural identity, never the bound numerical values. Assign distinct numerical binding slots to canonical logical factor occurrences `(source_term_ordinal, adjoint_component, factor_position)` and retain `source_term_id` separately as provenance; the ordinal indexes the explicitly preserved canonical specification term sequence, not inferred scientific identity. Current `QuantumLatticeSpecification` refuses duplicate term IDs; preserve that admission rather than accepting invalid duplicate terms. Same local labels, structural factor IDs, object coincidences, or initially equal values are not a shared numerical-binding contract. Keep the exact term/order-to-slot mapping through refresh so equal initial bindings can later diverge independently; adjoint bindings are derived from canonical source expansion, not independently editable values. Reuse a numerical leaf across logical slots only when an explicit immutable shared-binding contract also governs refresh. Monomial factor references use separate topology/binding indices and prepared site/parity metadata; do not retain dense backup matrices.

Use existing monomial expansion and ordered physical factors. Do not add sparse fields to `CompiledMonomial` or change `PreparedQuantumLattice` leaf order. The new prepared target retains sparse tables, coefficients, recipe metadata, and semantic IDs; do not keep another dense local-matrix copy in its device runtime tree merely as a canonical backup. Original local matrices are host preparation inputs owned by the caller; release internal preparation temporaries after binding the sparse runtime representation.

Sparse admission uses the product of prepared nonzero-column width bounds, not product local dimensions. Bound transition-table bytes, factors, monomials, raw routes, coalesced column targets, decoded-coordinate workset, and scratch separately. A large local dimension with narrow raising/lowering support may be admitted; a genuinely dense local matrix must still meet its explicit route/workspace budget.

Refresh is a native host binding operation before a run, not replanning inside iteration. It requires exactly the same structural transition support and physical domain; changing support requires preparation of a new artifact. A numeric refresh changes operator/prepared identity and invalidates an existing projector continuation/history. Starting a new problem from a previous wavefunction is explicit initialization, not silent continuation under a new Hamiltonian.

Column-action and refresh regressions must include two same-labelled local operators with the same structural factor ID/support and different diagonal values, placed in valid distinctly identified terms, plus initially equal logical bindings that diverge on refresh. Independently verify the sum of both terms; a topology cache must not substitute one numerical matrix for the other. Invalid duplicate-term-ID specifications continue to refuse. Numeric prepared/operator identity fingerprints every binding in its canonical logical slot order, and refresh verifies the original term identities/order and adjoint derivation.

### 3.4 Shared factor mathematics and true column orientation

Refactor only the existing canonical predecessor/parity operation needed by both dense and sparse actions. Preserve rightmost-first factor traversal, evaluate fermionic parity on the current intermediate configuration before each factor, and preserve coefficient/local-factor multiplication order. Repeated-site factors must work.

`QuantumColumn` is an exact bounded diagnostic/application result with diagonal, packed targets, coalesced `H[target, source]` values, masks, raw-route/unique-target/cancellation counts, and refusal evidence. It never advertises an `ArraySpace` dimension or dense materialization capability. It is not a reinterpretation of `ConnectedConfigurations`.

Exact outgoing columns enumerate sparse raw routes, group by exact packed target, and sum before removing zeros. The diagonal sums every route returning to the source. It is not the product of local diagonals: repeated-site off-diagonal factors can compose to a diagonal action. Computing this exact diagonal may require the full bounded sparse route traversal; retain that cost in work evidence and benchmarks.

The exact propagation stream can emit raw route contributions directly to the step accumulator, avoiding an unnecessary separate per-source coalescing allocation. Its final same-target sum must equal the coalesced column. Both paths share the same native route walker.

### 3.5 Genuine raw-route proposal

`sample_raw_excitation` selects a monomial with a documented prepared probability and then one valid local transition per reversed factor. The initial implementation uses fixed uniform monomial selection and uniform selection among structurally valid local transitions for the current local input. Each selection uses a distinct semantic factor/attempt role. Return:

- The packed target and raw route matrix element, including coefficient and fermion phase.
- The product proposal probability `p_route`, or its checked log representation with a representable positive probability for depositing a nonzero event.
- Route identity, validity, and whether the route is off-diagonal.

A dead local transition or final diagonal route is a null spawn attempt; it is not resampled. Diagonal routes are already included in the exact diagonal update. A constrained-domain violation is a failure, not a null physical transition.

Probabilities refer to raw paths, not coalesced destinations. Duplicate paths and cancellation are legal. The governing identity is `sum_r p_r * (h_r / p_r) = sum_r h_r`. A source with `n` attempts deposits `-dt * h_r * c_source / (n * p_r)` for each off-diagonal sampled path. Invalid/zero paths contribute no value without conditioning the proposal on acceptance.

The sampled method must walk one path; it must not call `outgoing_column` for every event or advertise heat-bath acceleration while always enumerating the complete column. Exact diagonal traversal can still dominate some models; measure and state that honestly. Heat-bath and without-replacement policies are not part of this plan.

## 4. Shared sparse extension and clean scalar compatibility

### 4.1 Generalize `KeyGroupPlan`, not the class hierarchy

Change the existing `key_upper_bound` contract to `int | tuple[int, ...]`:

- Integer bound: existing scalar keys, shapes, IDs, fingerprint payload, scalar error behavior, and lookup/alignment remain unchanged.
- Nonempty tuple of bounds: keys have shape `case_shape + (item_capacity, word_count)`; masks remain `case_shape + (item_capacity,)`. Bounds apply word by word. Tuple width and ordering participate in vector-plan identity.

Do not add an empty layout field to scalar fingerprint payloads or change the existing field count just for an optional vector shape. Distinguish scalar and vector key domains through the existing bound value and exact validation. Preserve current scalar numerical leaves and static values.

Use explicit invalid flags for vector sorting, not `maximum_uint32 + 1` sentinels. Sort by validity, the complete word tuple, stable event ID, and final original-index tie-break. Equality/group boundaries compare every word. Lookup is exact lexicographic search over compact group keys, not hash equality or a dense reverse map. `align_key_groups` must support both representations using the same lookup contract.

Stable event IDs remain scalar integers in this change. The projector admits per-step total event ordinals within signed-int64 range before execution; it does not need another multiword stable-ID API. Physical randomness uses full address words, not these reduction-order ordinals.

Language-server evidence: `KeyGroupPlan` had 46 references across native consumers. `key_upper_bound` had six references, including one external scalar assumption in `topology/_components.py::ComponentTransitionPlan.key_dtype`. Update that scalar owner to validate/narrow the bound without a cast/assert/new metadata field. Do not convert topology pair keys to word vectors.

### 4.2 Seeded signed reduction is required for bounded streaming

Add `reduce_key_groups` over `KeyGroupState`, using the canonical existing reduction owner. `KeyGroupAccumulation` carries per-group high values and, for compensated mode, correction values. The reducer accepts an optional seed in the current output-key layout and a value-active mask when key-retention entries do not represent numerical events.

Factor/reuse the current `_sum_segments` algorithm rather than writing a solver-local segment sum. Preserve existing relation reduction behavior: zero seed, current order, and final high-plus-correction produce the same result as before.

For streaming projector events:

1. Merge old accumulator keys and the new event keys using the same native grouping/lookup.
2. Align the existing high/correction values to the new compact group layout.
3. Add individual new contributions in their declared per-target order to those seeds.
4. Carry high and correction separately to the next chunk.
5. Collapse high plus correction only after all step contributions have been added.

Reducing a chunk to a subtotal and then adding the subtotal is prohibited for the deterministic replay route: it changes floating-point association with chunk width. Adding padding or seed-only entries must not alter the accumulator. Mask invalid NaN padding before arithmetic; do not suppress genuine nonfinite active values.

Existing `EdgeRelation`/`RowRelation` remain scalar int32 route structures. Do not encode a multiword configuration through a fabricated dense target size or scalar route index.

## 5. Projector problem, resources, execution, and RNG

### 5.1 Scientific problem and policy

The problem binds one time-independent certified self-adjoint original Hamiltonian, one endomorphic configuration domain, a fixed sparse physical initial vector, an optional frozen guide, a fixed physical trial vector, requested observables, explicit unit/provenance identity, and a constant positive finite inverse-energy step `dt`. Use the native unit metadata owner; do not infer physical units from coefficient dtype or silently equate a dimensional time with inverse energy. State the scaled imaginary-time convention explicitly.

Replicas start from the same deterministic physical initial vector and then use distinct semantic random streams and separate shift controllers. Sharing a deterministic starting vector or frozen guide does not introduce evolving stochastic dependence. Do not add automatic phase alignment, resampling between replicas, shared population feedback, or excited-state orthogonalization.

Admit `exact`, `sampled`, and `semistochastic` spawning. In semistochastic mode, use a declared raw-route upper bound, not a falsely advertised coalesced nonzero count, for the relative criterion. Choose exact application when `boost * abs(c) >= raw_route_bound * relative_threshold` or `boost * abs(c) > absolute_threshold`. The relative comparison is inclusive; the absolute comparison is strict. Implement comparisons without avoidable overflow. Record which count/bound drives the decision.

The relative raw-route bound is derived from the unchanged physical transition topology, not the configured maximum route/column capacity. Increasing resource limits must not change that bound, exact-versus-sampled choices, proposal probabilities, or the scientific sampler identity.

For sampled sources, `n = max(1, ceil(boost * abs(c)))`, subject to explicit checked attempt/work limits. If `n` exceeds the admitted limit, refuse; do not clamp `n`, drop attempts, or silently switch scientific policy to fit storage. Exact diagonal action is always included once.

Compression is either absent or one final stochastic threshold projection with finite positive `theta`. For nonzero `z` with `abs(z) < theta`, retain `theta * z/abs(z)` with probability `abs(z)/theta`, otherwise zero; leave larger values unchanged. Draw once per complete target coefficient. This projection preserves the conditional vector expectation; it is not a proof of an unbiased finite-population energy.

Controller modes are fixed shift and double-log population feedback. For nonzero finite next represented population `N_next`, update the latter using the declared damping/restoring terms and target population. Extinction is detected before any logarithm. Population is `sum(abs(represented_coefficients))`; target population, occupied configurations, event count, and physical norm are distinct. Record applied and next shift semantics. No adaptive time step or retry loop is introduced.

For a Hermitian deterministic projector with shift at the targeted eigenvalue, the fixed-point stability condition involves `dt < 2/(E_max - E_0)`. Do not infer a certified bound for an unranked domain from a finite probe or a norm estimate. A supplied step is not a universal stability certificate. Qualification controls must study step dependence and retain instability/refusal evidence. The exact projector's eigenvectors do not acquire a deterministic Euler fixed-point bias merely because a finite step is used; population control and finite-history corrections are separate issues.

### 5.2 Resource dimensions and array contracts

| Symbol | Meaning | Representative runtime shape |
| --- | --- | --- |
| R | Replica count | Replica-leading dynamic state fields |
| S | Committed/final support capacity per replica | Keys `(R,S,K)`, coefficients/active `(R,S)` |
| K | Packed address word count | Uint32 key word axis |
| G | Intermediate distinct-target accumulator capacity, at least S | Keys `(R,G,K)`, high/correction/active `(R,G)` |
| E | Event chunk capacity | Targets `(E,K)`, values/valid/event IDs `(E,)` per lane/workset |
| A | Maximum sampled attempts per source | A refusal/work bound, not mandatory S-by-A retained storage |
| B | Exact column diagnostic target capacity | `(B,K)`/`(B,)` only when a column result is requested |
| C | Bounded source/decode workset | At most C decoded configurations live |
| T | Raw accepted-step history capacity | Per-replica records `(R,T)`; measurement records additionally have observable axes |
| O | Requested physical observable count | Explicitly admitted finite static set |
| P | Ordered distinct replica pairs, R*(R-1) | Pair records `(P,T,O)` and `(P,T)` |

Admission computes byte limits for tables, state, source/decode workset, E-sized events, G-sized high/correction accumulation, column diagnostics, physical contractions, raw histories, checkpoints, and compiled iteration work counters. Check products with host integers before allocating or narrowing. Do not allocate `S * (B + A)` full event banks or all support trajectories by default. No global Hilbert vector is allocated.

The step event stream has a fixed total order: canonical source key, diagonal event, then native exact route ordinal or sampled attempt ordinal. Chunking is scheduling only. Each source reads the previous committed vector throughout the step; in-place propagation of already updated coefficients is forbidden.

### 5.3 Complete step transaction

1. Validate state/problem/plan/domain/operator/guide/RNG compatibility and accepted-step/history capacity.
2. Reject an initially extinct/invalid support before route queries or random draws.
3. Start an empty intermediate accumulator. For each canonical source, emit `(1 + dt*(S_shift - H_ii))*c_i` once, followed by exact or sampled off-diagonal contributions.
4. Process fixed event chunks into G-sized seeded native accumulation. Do not cull small coefficients, discard temporarily cancelling groups, or compress between chunks.
5. If required intermediate groups exceed G, terminate with `INTERMEDIATE_GROUP_OVERFLOW`, even if later cancellation might reduce final support below S. G is a real working-set boundary, not an estimator approximation.
6. After every source/event is processed, form complete coefficients and remove exact zeros. Retain pre/post-annihilation norms and cancellation evidence.
7. Apply optional late threshold compression exactly once. Check final retained support against S; `FINAL_SUPPORT_OVERFLOW` is terminal refusal, not top-k truncation.
8. Detect post-compression extinction before controller logarithms. Validate finite next represented population/controller state.
9. Produce physical overlaps and one raw accepted-step record. Observable/guide numerical failures are failed candidates. A zero statistical denominator is retained as a measurement/qualification issue, not repaired into a finite ratio.
10. Commit support, controller, history, root/step cursor, and all replica states atomically. If any replica fails propagation, roll back the batch and terminate. An incomplete ensemble window cannot be reported as a completed qualified calculation.

Use the existing lifecycle transaction algebra. The input state remains reusable and undonated. Reuse internal scratch under explicit solver ownership; no donation of caller-owned state is implicit. Runtime loops use `lax.scan`/`while_loop`/`fori_loop` and bounded worksets; no Python iteration over runtime-sized support or host synchronization inside the step.

Status must distinguish admission/operator/guide failure, attempt/work limit, intermediate grouping, final support, nonfinite propagation/controller, extinction, and history exhaustion. Preserve a deterministic first-failure precedence and requested/used resource evidence. Estimator statistical status remains separate from propagation status.

### 5.4 Logical RNG and resource-only replay

Reuse `derive_key` with a stable semantic namespace/domain/operator sampling identity, accepted logical step words, replica identity, full source address words, and local route/attempt/factor role. Target compression uses target address words and a separate role. Fold large counters as exact uint32 words and refuse wraparound before narrowing. Do not include chunk width, source slot, sorting position, capacity ID, or wall-clock attempt count in physical keys.

A failed overflow does not advance the physical step or change physical randomness. There is no automatic retry. An explicit resource-only transport can increase capacities and replay the same failed logical step with the same draws and numeric policy. Retrying with fresh draws until a step fits would condition the process on successful capacity and is prohibited.

Reduction-order event IDs are bounded signed-int64 ordinals within a step; they are not the RNG identity. Canonical source/event order is fixed. Changing only chunk width must preserve deterministic/compensated addition association and final canonical state on the same backend. Arbitrary event-order changes are not promised bitwise equivalent. Backend changes need an explicitly qualified tolerance replay class, not a universal bitwise claim.

## 6. Frozen guides and physical sparse observations

### 6.1 Positive guide contract

`QuantumGuide` is domain-bound and captures a frozen immutable amplitude provider/snapshot with an explicit semantic/provider identity and numeric fingerprint. Existing neural models returning `LogAmplitude` can supply the magnitude; any mapping from local-state indices to model inputs must be explicit, not inferred (for example 0/1 occupations versus -1/+1 spins).

Use finite `log_abs` to evaluate the positive magnitude; discard phase only because this guide contract explicitly represents a positive magnitude. Default `reject` refuses nodes/nonfinite active evaluations. The optional `positive-log-floor` policy applies one declared finite lower log bound to valid nodes/small magnitudes, records how the actual guide was defined, and uses the same floored guide everywhere. Invalid NaN/+infinity is not repaired. An explicit nonzero floor defines an invertible diagonal similarity; it is not inherently a Hamiltonian-spectrum approximation.

Retain global guide nonzero/admissibility assumptions separately from encountered runtime checks. Checking only sampled addresses does not prove a callable is finite/nonzero over its entire represented domain. Finite table guides can be fully checked; a positive-by-construction frozen provider can carry a declared contract whose runtime violations refuse. Never promote empirical checks into a global spectrum certificate.

Compute ratios in log space and fail when a required represented ratio/metric/contraction is outside the admitted numerical range. No hidden jitter, support restriction, both-zero bypass, adaptive guide update, or live optimizer parameters. Changing the guide invalidates continuation and the estimator window; explicit new initialization is required.

### 6.2 Similarity and metric invariants

For `D = diag(g)` and `d = D c`:

- Guided outgoing action: `H_g[target,source] = g(target) * H[target,source] / g(source)`.
- Physical metric: `M = D^{-dagger} D^{-1}`.
- Physical observable action: `Q_physical_in_guided_frame = D^{-dagger} Q D^{-1}`.
- Physical ratio: `(d_a^dagger * Q_physical_in_guided_frame * d_b) / (d_a^dagger * M * d_b)`.

For the positive guide, `M_i = exp(-2 * log(g_i))` and observable entries are `Q_ij/(g_i*g_j)`. Document and independently test the general complex-conjugation identity `Q_ij/(conj(g_i)*g_j)` even though complex-phase guides are not a selector in this implementation. This prevents a wrong bra/adjoint convention.

A guided Hamiltonian is generally not Euclidean self-adjoint. Retain original-H evidence and matching metric identity; do not route it through `LocalHamiltonian` or a finite basis-scaling transform. Initializer accepts physical coefficients and deliberately transforms once to the represented coefficients; state fields explicitly name their representation/guide identity. Do not store both physical and represented coefficient arrays redundantly.

The inspected upstream guide undoer appears to use `value/(2*g)` instead of the documented `value/g^2`; this is a source-level concern, not a reproduced upstream defect. A native regression with `g=3` must recover metric `1/9`, and a two-address complex observable must recover the original physical contraction. No upstream wrapper is copied.

### 6.3 Physical overlap execution

Projected energy uses a fixed physical trial vector: retain `X = y^dagger H c` and `Y = y^dagger c` separately, applying inverse guide factors consistently. Do not use a mean of instantaneous `X/Y`.

Replica estimates use ordered distinct pairs `a != b`; retain raw pair numerator/denominator time series. At each time, aggregate pair sums and analyze the aggregate aligned stream. Pairs sharing a replica are not independent observations. For self-adjoint observables, Hermitian paired contractions may share work only under the certified property; a general complex observable must not have its imaginary component silently discarded.

Evaluate sparse contractions through original outgoing columns/streamed routes and exact native lookup in the bra support. Absent bra coefficients are genuine sparse zeros; invalid operator/guide evaluation is not. Avoid support-by-support Cartesian products and dense incidence matrices. Bound pair/observable work and scratch before execution. Replica propagation controllers and streams are independent; a common frozen guide is allowed. No statistical independence proof is inferred from replica IDs alone.

## 7. Correlated ratios and projector estimators

### 7.1 One covariance route

Choose synchronous complete batch-means covariance, reusing extracted free-energy sequence/group/block selection invariants. Do not add a second bootstrap, Jonsson M-test selection default, trace-only multivariate cutoff, or PSD clipping.

The generic ratio input consists of aligned numerator and denominator series with explicit stream/group and draw identity, one validity mask, and a declared sampling origin. Padding is allowed; dropping nonfinite active draws or independently masking numerator/denominator is not. Preserve stream boundaries; dependent streams must be synchronously grouped, not pooled as IID.

Represent real pairs as `(X,Y)` and complex pairs as `(Re X, Im X, Re Y, Im Y)`; use the smallest truthful representation for mixed real/complex inputs. Retain the complete covariance. Resolve one common block length conservatively from diagnostics of every varying real component and the estimated ratio-influence series. Permit an explicit block length only when resolution gates support it. Bound maximum lag, minimum draws, and minimum complete blocks; a cutoff that fails to resolve the relevant correlation is a refusal, not success with a short window.

Existing scalar initial-monotone diagnostics and free-energy initial-positive diagnostics remain distinct established algorithms. Extraction must preserve free-energy values, masks, grouping, exception order, block indices, IDs, and bootstrap results. Share selection/grouping kernels, not a silently changed statistical convention. Use host-only immutable selection metadata where the existing owner does, with explicit analysis boundaries; numerical covariance/ratio kernels use JAX. There is no device-to-host synchronization inside projector iteration.

The existing free-energy helper does not classify a still-positive correlation sequence reaching the lag limit as unresolved. Preserve that established free-energy behavior, but return additional window-termination evidence from the shared kernel and require a resolved positive-sequence cutoff for the new ratio owner. Exhausting the configured/available lag range with relevant positive correlations is `CORRELATION_UNRESOLVED`; an explicit block length does not bypass that gate. Add a manufactured positive-tail-at-maximum-lag refusal case.

For K equal-length retained blocks with block means `b_k`, let `b_bar` be their mean and compute covariance of the mean as `sum_k ((b_k-b_bar)(b_k-b_bar)^T)/(K*(K-1))`. This is PSD by construction. Retain block length/count and discarded incomplete-tail records. Handle independent groups with their explicit sample weights and covariance sums; dependent channels stay in one synchronous block vector. Do not apply an arbitrary eigenspectrum repair. Correlation resolution and the asymptotic Gaussian approximation remain evidence/assumptions, not a finite-sample theorem.

No single trace-averaged ESS replaces ratio uncertainty. Return component/influence temporal diagnostics and block evidence alongside the full mean covariance. Retain pair sums before this calculation so shared-pair cross-covariance is present without allocating a full pair-count-squared covariance matrix.

### 7.2 Ratio qualification and edge cases

Return means, an exploratory point ratio when defined, full mean covariance, ratio covariance/standard error when justified, denominator diagnostics, confidence evidence, and statistical status. Unsafe qualified values/intervals are explicitly unavailable; do not return a conventional finite SE with a success flag after a denominator gate fails.

For real ratios, use Fieller's quadratic `(x-r*y)^2 <= z^2*(s_xx - 2*r*s_xy + r^2*s_yy)` with the same joint covariance. Retain bounded/unbounded/disconnected/all-real/empty/singleton classifications and refuse an ordinary bounded estimate when the denominator confidence interval includes zero or requested magnitude/sign floors fail. Implement numerically stable boundary handling, including zero discriminant and degenerate coefficients; do not hide these in a generic optimizer.

For complex denominator means, do not invent a scalar Fieller interval. Require the origin to be excluded from the declared confidence region for `(Re Y, Im Y)` and satisfy the magnitude floor. A conservative enclosing confidence ball based on `trace(cov_Y)` is sufficient and avoids unsafe inversion of singular covariance. Propagate the full real joint covariance through complex division only after that gate. Label the interval/covariance approximation honestly.

An observed constant stochastic history is not proof of zero population uncertainty. Return `ZERO_VARIATION_UNRESOLVED` unless the source explicitly establishes deterministic records or an exact deterministic ratio relation. A deterministic nonzero denominator can produce a singleton/zero sampling covariance, but this does not certify eigenstate convergence. A zero denominator is always unsafe. Preserve the existing scalar diagnostic's zero-variance behavior.

### 7.3 Projector-specific statistics

The solver-owned estimator adapter consumes typed raw histories with original/domain/operator/guide/metric identities. It produces:

- A correlated shift mean for the explicitly recorded applied-shift series, separately flagging population-feedback assumptions/bias.
- Projected physical energy from aligned numerator/denominator histories.
- Physical replica Rayleigh/observable ratios from aggregate ordered-pair sums.
- Finite-history reweighted projected and replica estimates.

Keep propagation completion, statistical validity, finite-history/population systematic assumptions, finite-domain/truncation scope, and scientific qualification separate. A small SE is not a no-bias or no-sign-problem certificate. Absence of initiators and unbiased per-step compression are explicit algorithm properties; do not label compression itself intrinsically biased. Finite-population feedback, sign-resolution problems, hard support/resource truncation, and statistical ratio effects have different meanings.

### 7.4 Finite-history convention

Only exponential finite-history weights are implemented in this plan. Do not call a scalar product of linear factors an exact cancellation of the additive Euler projector, or add a second weight selector without a consumer.

Record the shift actually used for transition from state n to n+1. For state n and history depth h, use the h applied shifts immediately preceding that state: `log(w_h,n) = -dt * sum_{k=n-h}^{n-1}(S_k - E_ref)`. For replica pairs, multiply the two replica weights in log space. Define the indexing in one place and test it on manufactured histories.

Record applied shifts at every accepted step. Measurements may have a declared cadence, but missing per-step shifts cannot be fabricated from decimated output. Require complete h-step windows; either retain sufficient pre-burn-in history or exclude early measurements explicitly. Incomplete windows are not shortened silently. Carry all required history through checkpoint/continuation.

Use stable normalized log weights, retaining normalization scales and both ordinary weight ESS and temporal/block diagnostics. Constant weights, h=0, h=1, insufficient history, severe weight concentration, and numerical range failures have explicit outcomes. Finite h is not advertised unbiased. State only the applicable asymptotic limits and the bias/variance tradeoff; history-depth and population studies are qualification evidence, not error-bar subtraction.

For the exponential correction, any asymptotic bias-removal statement must explicitly include the applicable continuous-time limit `dt -> 0`, sufficiently long history `h -> infinity`, and a history horizon `h*dt` resolving the relevant physical relaxation, together with population/sign/stationarity assumptions. Infinite history alone does not exactly undo the additive finite-step projector. Qualification records both h and h*dt and varies history, population, and step separately; no universal order-independent joint-limit claim is made.

## 8. Lifecycle and result persistence

Checkpoint only at a fully committed boundary: complete event accumulation, annihilation, compression, physical observation, controller update, and history publication. Never archive pending event chunks or an incomplete accumulator as a restartable state.

Store canonical packed support, represented coefficients/masks, logical step/time, per-replica applied/next controller state and previous population, raw history/cursors, root key implementation/data, domain/codec/operator/prepared numeric IDs, guide/metric snapshot, scientific sampling/reduction policies, resource plan, precision, and replay classification. Serialize typed key data only at the existing archive boundary and restore the recorded key implementation explicitly; do not assume every typed key has the default implementation or expose a legacy uint32 key publicly.

Use existing `RuntimeCheckpointEnvelope`, verification/read/write functions, `RuntimeRestartRelation`, and durable lifecycle/artifact publication. New projector lifecycle functions are adapters binding these contracts, not a second ZIP/schema/repository implementation. Final raw/statistical results use existing `ResultManifest`/`ResultRevision` and lifecycle result storage where persistent results are requested. No format generation suffix or compatibility schema is introduced.

Restore requires the matching caller-prepared operator/domain/guide. Validate the complete restored Strict value once with `phydrax.typing.validate`, then check semantic bindings and scientific state invariants. Callables/providers are not deserialized from arbitrary payloads. Refuse changed operator values, domain/order/charges, guide, sampling/controller policy, precision, or RNG implementation without a scientifically appropriate new initialization.

The one supported shape transport enlarges resource-only capacities and pads inactive rows/history, preserving physical state, data, draw identity, addition order, and failed logical-step randomness. Its restorer verifies exact old/new scientific bindings and monotone capacities. Same-plan ordinary continuation and resource-only replay are distinct operations of the owning restart relation, not resampling policies.

Single-device scope is explicit: refuse non-addressable global arrays and unsupported multi-process ownership at admission. Existing distributed checkpoint publication does not supply global target-event exchange/annihilation. Do not reuse a PIC executor, all-gather full wavefunctions, silently serialize a distributed request, or claim distributed replay. Distributed projector execution needs a separate destination-ownership/collective/load-balancing design and qualification.

## 9. File-by-file implementation map

### 9.1 Package implementation

| Path | Action | Responsibility / acceptance |
| --- | --- | --- |
| `phydrax/operators/quantum/lattice/_address.py` | **NEW** | Packed codec, domain/address validation, exact rank-free charge predicates, scientific IDs, nominal dimensions. No dense/global basis or duplicate coordinate state. |
| `phydrax/operators/quantum/lattice/_column_compile.py` | **NEW** | Sparse transition tables/monomial recipes, column resource policy, sparse preparation and same-support refresh. Reuse canonical term expansion; no retained dense backup matrices. |
| `phydrax/operators/quantum/lattice/_column.py` | **NEW** | Exact streamed/coalesced columns, exact diagonal traversal, single raw-route proposal, original-H/domain evidence, and finite refusal records. Uses native sparse grouping/reduction and shared parity action. |
| `phydrax/operators/quantum/lattice/_guide.py` | **NEW** | Frozen positive guide, explicit node/floor policy, guided outgoing action and physical inverse metric. No live optimization or Euclidean-Hermitian claim. |
| `phydrax/operators/quantum/lattice/_operator.py` | Edit narrowly | Share predecessor/parity/factor operation needed by sparse and dense routes; change the private predecessor helper to consume canonical specification metadata if needed. Preserve public dense route outputs/order/admission. LSP found only its definition and one existing call. |
| `phydrax/operators/quantum/lattice/_compile.py` | Edit only if needed | Expose a dependency-minimal internal expansion/structural preparation invariant to the sparse target. Do not add fields to existing prepared/monomial classes or route existing consumers through the new target. |
| `phydrax/operators/quantum/lattice/__init__.py` | Edit | Explicitly export address/column/guide preparation and types once; keep existing names. |
| `phydrax/sparse/_key_groups.py` | Edit | One scalar/word-vector grouping, validation, exact lookup and alignment implementation; preserve scalar shape/identity contract. |
| `phydrax/sparse/_execution.py` | Edit | Seeded high/correction accumulation and compact key-group reduction. Reuse existing `_sum_segments`; preserve relation reductions. |
| `phydrax/sparse/__init__.py` | Edit | Explicit grouped-reduction exports only; no second multiword family. |
| `phydrax/topology/_components.py` | Edit narrowly | Own scalar narrowing in `ComponentTransitionPlan.key_dtype`; preserve pair-key arithmetic, IDs, and transition semantics. |
| `phydrax/uq/_correlation_selection.py` | **NEW, private shared owner** | Extract exact grouped sequence/window/block construction invariants from free energy; add complete-block metadata needed by ratios without changing existing free-energy convention. |
| `phydrax/uq/_free_energy.py` | Edit narrowly | Replace extracted private machinery with its canonical shared owner; retain public records, numerical behavior, grouping, IDs, bootstrap, and exceptions. No estimator rewrite. |
| `phydrax/uq/_correlated_ratio.py` | **NEW** | Aligned real/complex batch-means covariance, ratio influence diagnostics, Fieller/complex-denominator gates, deterministic/zero-variation distinction and native status. |
| `phydrax/uq/__init__.py` | Edit | Export only the public ratio API; correlation extraction stays internal. |
| `phydrax/solver/_projector_monte_carlo_contracts.py` | **NEW** | Problem, capacities/policies, typed state/history/step/result/evidence, portable status, unit/provenance and compatibility validation. |
| `phydrax/solver/_projector_monte_carlo_step.py` | **NEW** | Canonical event scheduling, seeded accumulation, late compression, independent controller updates, atomic accepted-step boundary. Split helpers by emission/accumulation/compression/controller invariants, not arbitrary line counts. |
| `phydrax/solver/_projector_monte_carlo_observables.py` | **NEW** | Sparse physical trial/replica contractions with original-column orientation and matching guide metric; bounded ordered-pair records. |
| `phydrax/solver/_projector_monte_carlo_estimators.py` | **NEW** | Raw-history selection, exponential finite-history weights, projected/shift/replica producers and separate systematic claim records. Compose the generic UQ ratio owner. |
| `phydrax/solver/_projector_monte_carlo_lifecycle.py` | **NEW** | Native checkpoint/result adapters, typed-key archive boundary, full restore validation, resource-only shape restorer. No new persistence engine. |
| `phydrax/solver/_projector_monte_carlo.py` | **NEW** | Prepare/initialize/step/solve orchestration with stable compiled callable identity, immutable caller state, history-capacity admission, and continuation. |
| `phydrax/solver/__init__.py` | Edit | Explicit canonical projector public names and no forwarding aliases. |
| `phydrax/operators/quantum/lattice/_qualification.py` | Edit | Exact bounded address-column/projector/guide candidate tuples and gates. Preserve existing tuples; do not label research controls released. |
| `phydrax/applications/_condensed_matter_evidence.py` | Edit if required by aggregate evidence | Retain the new owner profiles and campaign references once in the existing condensed-matter ledger, with exact candidate scope. No unrelated promotion. |

Intentionally unchanged package owners: `_model.py`, `_sector.py`, `_vmc.py`, `_strings.py`, quantum `_discrete.py`/`_amplitude.py`, finite linalg/eigen, tensor-network lowerers, existing VMC/TDVP/TPQ/response/jump solvers, `_sampling/_addressing.py`, `_posterior_reweighting.py`, `_ensemble_reweighting.py`, and core lifecycle formats. Docstrings may receive a necessary orientation cross-reference, but not an API migration or numerical rewrite.

The implementation phase must refresh LSP references before any exported signature or private-helper cutover; the plan's current reference inventory is a baseline, not permission to miss intervening callers.

### 9.2 Tests and scientific fixtures

| Path | Action | Consumer-visible contracts |
| --- | --- | --- |
| `tests/_support/projector_monte_carlo.py` | **NEW** | Tiny independently specified boson/fermion/complex controls and bounded host enumeration/reference helpers; no copy of native route physics. |
| `tests/unit/operators/quantum/test_quantum_address.py` | **NEW** | Cross-word exact round trips; legitimate zero key; invalid spare bits/ranges; different physical identities; fixed/mixed charge predicates; no scalar-rank requirement. |
| `tests/unit/operators/quantum/test_quantum_columns.py` | **NEW** | Independent exact diagonals/columns; repeated-site paths; duplicate cancellation; complex orientation; CAR signs; genuinely sampled path probabilities and exhaustive conditional expectation; sparse versus genuinely dense admission; equal structural IDs with unequal numerical bindings; initially equal bindings that diverge on same-support refresh; topology-change refusal. |
| `tests/unit/operators/quantum/test_quantum_guides.py` | **NEW** | Node/floor policy; frozen binding; original-H/metric compatibility; g=3 metric 1/9; complex physical observable recovery; numerical-range refusal; no global certificate from sampled checks. |
| `tests/unit/test_sparse_substrate.py` | Extend targeted cases | Preserve scalar grouping/reduction; vector same-prefix/different-tail equality/lookup/alignment; masked NaN; wide uint32 keys; seeded complex cancellation; chunk-width invariant high/correction; independent capacity failures. Do not refactor the unrelated suite. |
| `tests/unit/topology/test_components.py` | Extend one boundary scenario | Narrow/wide scalar pair-key transitions produce correct logical component IDs/overlap, including pair keys beyond int32; do not merely assert a dtype field. |
| `tests/unit/uq/test_correlated_ratio.py` | **NEW** | Joint cross-covariance changes ratio uncertainty; complete block/group boundaries; correlated influence selection and positive-tail lag-exhaustion refusal; real Fieller boundary classes; singular/complex denominator-origin gate; constant stochastic versus deterministic data; invalid active masks; insufficient history/blocks. |
| `tests/unit/uq/test_free_energy.py` | Existing regression, edit only if necessary | Shared selection extraction preserves existing analytic/bootstrap estimator behavior, dependence, and errors; no source-text/copy tests or retuned expected values. |
| `tests/unit/solver/test_projector_monte_carlo.py` | **NEW** | Exact physical one-step update; sampled conditional expectation; semistochastic equality boundaries; all-chunk cancellation before compression; storage/chunk-independent RNG; separate intermediate/final/attempt failures; extinction before controller logs; batch rollback; immutable caller state; unsupported execution/original-H refusal. |
| `tests/unit/solver/test_projector_monte_carlo_estimators.py` | **NEW** | Ratio-of-means rather than mean-of-ratios; physical guided overlaps; aggregate ordered-pair covariance; incoming-shift indexing; h=0/1/full-window/weight concentration; separate statistical/systematic outcomes. |
| `tests/unit/solver/test_projector_monte_carlo_lifecycle.py` | **NEW** | Resume versus uninterrupted output; nondefault typed-key implementation; changed scientific binding refusal; history retention; complete restored validation; resource-only enlargement replays the failed step with the same draws; durable last-commit preservation. |
| `tests/integration/test_projector_monte_carlo.py` | **NEW** | One complete bounded public prepare/initialize/solve/analyze/checkpoint/resume workflow using independent bosonic energy/observable reference and an explicit guide. No broad portfolio wrapper. |
| `tests/typing/cases/projector_monte_carlo.py` | **NEW** | Public inferred state/column/ratio types and genuinely distinguishable wrong argument classes, closed selectors, and return contracts with exact line-local expected diagnostics. JAX tensor forms and `PRNGKey` erase to `jax.Array` statically: dtype, rank, typed-versus-ordinary array key, and same-class scientific-domain mismatches belong in runtime Strict/admission tests, not impossible static negative expectations. |
| `tests/typing/installed/consumer.py` | Extend existing fixture | Exercise the new exported constructors/results and truthful inferred types through installed-wheel imports. The installed checker discovers this directory, not the new source-tree case file; retain existing consumers. |
| `tests/unit/test_contract_declarations.py` | Existing gate | Runtime-resolvable native contract annotations for every new Strict module; no new source-text declaration tests. |

Use `strict_jax` where dtype/rank/warning policy is actually owned. Parametrize genuine scientific matrices/edge cases with diagnostic IDs. Keep failures with independent ownership separately collectable. Do not introduce loops/soft assertions to reduce item counts. Permanent tests must catch a plausible wrong sign, missing probability, bad covariance, partial commit, key collision, or lost history—not imports, forwarding, mock echoes, incidental defaults, or nonempty output.

Independent references:

- Two bosons on two sites: in basis `(2,0),(1,1),(0,2)`, hand-specify the matrix with diagonal `(U,0,U)` and hopping `-sqrt(2)*t`; reference ground energy is `(U - sqrt(U*U + 16*t*t))/2`.
- Few-mode fermions: host exterior-product occupancy removal/insertion with independently counted permutation signs; include a spectator between nonadjacent hopping modes.
- Complex three-site flux: independently specified ring matrix/Fourier spectrum and complex bra/ket contractions; catches accidentally real-only row/column tests.
- Duplicate paths: two independently specified terms reaching one target with exact signed cancellation, plus repeated-site ladder composition returning to the source.
- Stochastic unit tests: enumerate the finite proposal/compression outcomes and their theoretical probabilities against an independent one-step matrix action. Do not use a flaky single-run energy tolerance as a unit oracle.
- Statistical tests: finite manufactured paired/block records with an independently calculated joint covariance; a cross-covariance-sensitive ratio; signed near-zero denominators; slow correlated influence despite apparently benign individual channels; positive correlation tails exhausting maximum lag; degenerate real Fieller and singular complex-denominator covariance; manufactured h=1 incoming-shift records.

Qualification randomness is separate from deterministic unit checks. Perform deliberate-fault checks on the new shared scientific helpers: remove fermion sign, omit inverse proposal probability, compress per chunk, replace joint covariance with diagonal covariance, and replace inverse guide metric by 1/g. Each relevant contract must fail. Do not leave mutation scaffolding in the repository.

### 9.3 Benchmarks, qualification, examples, and docs

| Path | Action | Deliverable |
| --- | --- | --- |
| `benchmarks/projector_monte_carlo.py` | **NEW** | Phase-separated address/transition/column/step/observable/statistics/checkpoint campaigns using `benchmarks._runtime`; bounded scaling and matching-error comparisons. |
| `benchmarks/projector_monte_carlo.json` | **NEW, generated after execution** | Canonical measured records with source/environment/backend/precision identities, raw timing distributions, compiler bytes, logical retained bytes, and scientific error/bias evidence. No fabricated or empty result artifact. |
| `tools/sparse_execution_benchmarks.py` | Extend separate cases | Word-vector grouping and seeded signed cancellation; preserve existing scalar records. |
| `benchmarks/cm_quantum_lattice.py` | Existing affected control | Run finite compile/action/lowering controls after shared parity refactor; no wholesale driver rewrite. |
| `tools/projector_monte_carlo_qualification.py` | **NEW** | Locked finite controls, raw estimator histories, population/history-depth/step studies, refusal/replay evidence, and native campaign/reference IDs. Candidate-only evidence, not an automatic signed release. |
| `examples/projector_monte_carlo.py` | **NEW** | Real public boson problem, stochastic/semistochastic execution, optional frozen guide, joint estimator status, checkpoint/resume; no fake provider or full-Hilbert fallback. |
| `docs/guides_projector_monte_carlo.md` | **NEW** | Scientific conventions, resource envelopes, raw-route probabilities, late compression, guide metrics, statistical/systematic limits, lifecycle, and unsupported scope. |
| `docs/cookbook/projector_monte_carlo.md` | **NEW** | Minimal complete user workflow and overflow/replay/denominator handling. |
| `docs/api/solver/projector_monte_carlo.md` | **NEW** | Canonical projector APIs and status/replay/units. |
| `docs/api/uq/correlated_ratios.md` | **NEW** | Aligned series, common blocks, covariance and Fieller/complex refusal, deterministic versus unresolved constant data. |
| `docs/quantum_lattices.md` | Edit | Explain rank-free packed columns beside bounded direct sectors; preserve existing finite API and candidate claims. |
| `docs/api/operators/quantum.md` | Edit | Address/column/guide APIs and outgoing versus row orientation. |
| `docs/guides_quantum.md`, `docs/cookbook/quantum_vmc.md` | Edit narrowly | State that the new coefficient projector is not VMC/TDVP/jumps, and retain row local-energy semantics. |
| `docs/guides_uncertainty.md`, `docs/api/uq/index.md` | Edit | Link joint correlated ratios and distinguish temporal, weight, and systematic evidence. |
| `docs/guides_solver.md`, `docs/guides_solver_evidence.md`, `docs/guides_condensed_matter_production_evidence.md` | Edit narrowly | Projector lifecycle, bounded candidate evidence and explicit nonclaims. |
| `mkdocs.yml` | Edit | Add guide/cookbook/API navigation adjacent to quantum/VMC/UQ sections. |
| `CHANGELOG.md` | Edit after runtime proof | Record the complete native feature, precision/single-device scope and scientific nonclaims. |
| `docs/data/public_api.json` | Regenerate | Run the existing public API generator after integrated exports. |
| `docs/data/capabilities.json`, `docs/api/capabilities.md` and generator-owned portfolio/closure records | Regenerate | Run existing capability generator; retain actual generated changes, including aggregates affected by new candidate profiles. Do not hand-edit generated JSON. |
| `NOTICE`, `LICENSES/RIMU-MIT.txt` | Conditional only | Needed if substantial code is translated/reused; preserve the upstream MIT notice. Algorithmic independent implementation still cites its scientific references in docs. Do not add attribution boilerplate falsely claiming copied source. |

The existing `_builtin_catalog.py` profile provider already discovers `quantum_lattice_candidate_profiles`; no new registration convention is required. Confirm generated discovery, and edit registration only if actual canonical discovery requires it. Do not broaden source/closure taxonomies, provider adapters, or signed release records just to attach a paper citation.

## 10. Verification selection, smoke proof, and performance acceptance

All commands in this section are future implementation verification. None were run while creating this plan.

### 10.1 Minimal conservative test selection

At implementation time, combine this touched-surface map with changes since the last merge into `dev`; retain newly affected work by other agents. Do not run unrelated providers or the entire suite for this feature. Run affected tests with `-n auto`, except a truly isolated provider/device-topology invocation.

Required new tests are the address, column, guide, ratio, projector, estimator, lifecycle, integration, and static fixture paths in section 9.2. Required existing controls are:

- `tests/unit/test_sparse_substrate.py` and `tests/unit/topology/test_components.py` for shared key/reduction behavior.
- `tests/unit/operators/quantum/test_lattice_compiler.py`, `test_sector_operator.py`, and `tests/unit/operators/test_quantum_discrete.py` for shared factor/parity behavior and retained row convention.
- `tests/unit/uq/test_correlated_observable.py` and `test_free_energy.py` for unchanged scalar diagnostics and extracted selection.
- `tests/unit/discretization/test_particle_cell_list.py`, `test_pic_cell_binning.py`, `tests/unit/discretization/spatial/test_sparse_blocks.py`, `test_sparse_voxel.py`, and `tests/unit/threshold_dynamics/test_sparse_route.py` for real scalar grouping/lookup/alignment consumers.
- `tests/unit/applications/two_phase_flow/test_bubble_markers.py`, `test_bubble_coalescence.py`, the phase-field active-storage scenario in `test_phase_field_workflows.py`, and the halo-finder scenario in `test_cosmology_halos.py` when their grouping kernel path is changed; select the actual owning node IDs after reading collection definitions.
- `tests/unit/test_contract_declarations.py` and the pinned static fixture gate.

If vector support is a proven isolated new branch and existing scalar code is untouched, existing consumer coverage can be reduced to the scalar grouping/reduction and topology controls plus one real lookup/alignment and one case-batched consumer. Record that evidence-based choice. If the common sort/search/reduction kernel is refactored, all listed scalar consumer scenarios are affected. Do not silently skip them or expand to unrelated surrounding application suites.

No whole-suite fixture/configuration/test-architecture change is planned. If implementation introduces one, explicitly reassess the test scope under repository policy instead of treating this list as permission to miss it.

Use the configured formatter/linter, `python tools/check_typing.py check`, `python tools/audit_selectors.py`, existing public/capability manifest generation/check commands, the declaration/static fixture checks, and installed-wheel typing verification for changed public annotations. These tooling checks are not scientific smoke proof. Measure touched ordinary helpers before/after with configured/available cyclomatic, cognitive, NLOC, and parameter metrics; keep the repository's touched-function thresholds and decompose by invariant. Do not invent another typing/formatting convention.

### 10.2 Runtime smoke and locked scientific qualification

Exercise the actual public example/qualification program, not only unit kernels:

1. Prepare an unranked domain and sparse columns for the independent two-site boson control.
2. Initialize a fixed physical vector with a typed root key and at least two replicas.
3. Run bounded semistochastic propagation with both exact and sampled sources exercised; observe complete history and operational/statistical statuses.
4. Analyze projected and physical replica ratios with their denominator/block evidence.
5. Run the guide control and verify the physical reference/metric identity, not the Euclidean guided Rayleigh quotient.
6. Checkpoint and resume; compare against uninterrupted accepted-step output on the same backend.
7. Deliberately exhaust an intermediate capacity; observe preserved last state and explicit resource-only same-draw replay after enlargement.
8. Exercise a 100-mode rank-free address/column query under explicit small support/work budgets to prove that the changed surface does not require a globally rankable basis. Do not claim the resulting small run solves that model.

Qualification uses predeclared controls, seeds, populations, history depths, burn-in, draw counts, error thresholds and run counts. Retain all failed/poor-overlap results; no repeated fresh seeds until one passes. Separate stochastic uncertainty, population/history systematic studies, sign-resolution observations and finite local cutoff scope. A finite energy is not a release or generic sign-cure claim. Keep provider/device topology and precision identity in every record.

Suggested future invocations, with new driver arguments implemented as documented:

```text
JAX_ENABLE_X64=1 python examples/projector_monte_carlo.py
JAX_ENABLE_X64=1 python tools/projector_monte_carlo_qualification.py --output <evidence-path>
JAX_ENABLE_X64=1 python -m benchmarks.projector_monte_carlo --output <benchmark-path>
python tools/generate_public_api_manifest.py
python tools/generate_capability_inventory.py
python tools/check_public_api_manifest.py
python tools/check_capability_consistency.py
python tools/check_typing.py check
python tools/audit_selectors.py
python -m tools.check_installed_typing
```

Use the active pinned environment; do not create a second dependency set. The new benchmark output argument and example behavior are explicit additions, not claims that those drivers already exist. Select a concrete affected pytest invocation from the above path map after integration; no future generated node ID is asserted to exist now.

### 10.3 Phase-separated performance campaign

Use `benchmarks._runtime` for environment identity, host preparation, synchronized execution, lower/compile separation, compiler analysis, and logical bytes. Stable compiled callables are module-level; no runtime `jax.jit` construction.

Measure separately:

- Codec/charge metadata preparation and sparse transition/table binding/refresh.
- Exact diagonal, exact coalesced column, raw-route proposal, and raw event emission.
- Word-key grouping/lookup and seeded deterministic versus compensated reduction.
- Complete cold/warmed projector step and multi-step scan.
- Physical trial/replica observations, ratio analysis, finite-history weighting, and checkpoint I/O.
- Lowering, compilation, first synchronized execution, warmed distributions, compiler temporary/output/code bytes, argument bytes and logical retained bytes.

Vary the controlling capacities: site/address width, local transition sparsity, raw route bound, S, E, G, A, replica count, observable count and T. Include narrow large local matrices and genuinely dense local matrices; same-prefix/different-tail keys; high cancellation and uneven target occupancy; chunk widths that preserve association; prepared versus cold binding. Compare exact, sampled and semistochastic paths at matched scientific error/bias targets and disclosed resource limits. Do not compare unlike estimators or report only steps/sec as scientific efficiency.

The sampled path must not allocate complete outgoing columns per event, retain a dense global/canonical operator, or store an S-by-A bank. Benchmark scratch must scale with the declared workset and G/S/history sizes, not Hilbert dimension or an undeclared Cartesian product. Sorting/lookup every chunk has a real cost; retain it and do not claim accelerator superiority without measurements. Do not set a speculative speedup threshold against Rimu's CPU/MPI results.

After proof, audit remaining numerical Python loops, host synchronizations, full sorts, dense materialization, repeated preparation, and redundant state copies. Each remaining hit requires a bounded/static, preparation, output-size, or external-boundary rationale. Remove throwaway smoke/mutation scaffolds; retain only consumer-visible regression tests and requested benchmark/qualification records.

## 11. Completion gates

Implementation is complete only when all are true:

- Addresses distinguish scientific domains and round-trip exact multiword keys without rankability, sentinel collisions, or narrowing overflow.
- Native sparse column physics handles complex parity, duplicates and repeated-site diagonals; raw-route probabilities reproduce the independent one-step action.
- Every current scalar key-group consumer keeps its documented behavior/IDs; the widened scalar-bound reader is migrated and static typing is clean.
- Chunked propagation performs full annihilation before one compression, preserves declared addition order and physical RNG under chunk/resource changes, and exposes every refusal without clipping or fresh-retry conditioning.
- Physical projected and replica estimators use matching original/guide metrics, joint covariance and denominator safety; finite-history/population assumptions are explicit.
- Same scientific problem checkpoint/continuation and explicit capacity-only replay are demonstrated; incompatible scientific bindings and unsupported distributed execution refuse.
- The actual public stochastic workflow, independent finite controls, resource refusal paths, and phase-separated scaling campaign have been exercised with retained evidence.
- Existing finite-sector/VMC/statistics behavior is preserved by the selected regression controls, public annotation checks report zero diagnostics, and all affected exports/docs/examples/generated records agree.
- No unfinished selector, stub, fake fallback, obsolete alias, new schema generation, misleading acceleration name, or unqualified scientific release claim remains.

## 12. References and design-review disposition

- [Rimu source revision](https://github.com/RimuQMC/Rimu.jl/tree/869e334c101c82ca8640e54338c6ffa170886ec8).
- [Rimu software paper, algorithms/statistics/data structures/benchmarks](https://arxiv.org/html/2601.19505).
- [Column interface](https://github.com/RimuQMC/Rimu.jl/blob/869e334c101c82ca8640e54338c6ffa170886ec8/src/Interfaces/hamiltonians.jl).
- [Semistochastic and compression policies](https://github.com/RimuQMC/Rimu.jl/blob/869e334c101c82ca8640e54338c6ffa170886ec8/src/StochasticStyles/styles.jl).
- [Parallel accumulation phases](https://github.com/RimuQMC/Rimu.jl/blob/869e334c101c82ca8640e54338c6ffa170886ec8/src/DictVectors/pdworkingmemory.jl).
- [Statistical estimator documentation](https://github.com/RimuQMC/Rimu.jl/blob/869e334c101c82ca8640e54338c6ffa170886ec8/docs/src/statstools.md).
- [Guide metric source concern](https://github.com/RimuQMC/Rimu.jl/blob/869e334c101c82ca8640e54338c6ffa170886ec8/src/Hamiltonians/GuidingVectorSampling.jl).

Binding integration decisions already incorporated: one generalized key-group owner, packed address storage without dense backup, raw-route rather than mislabeled target probabilities, seeded accumulation across bounded chunks, overflow termination and same-draw resource replay, batch-means covariance without PSD clipping, no independence assumption for shared replica pairs, deterministic-versus-observed-constant distinction, explicit invertible guide floor, and no unrelated TPQ/RNG/format changes.

Independent source-only plan review identified and resolved two blocking design issues: structural local-operator IDs are not numerical binding IDs, and erased JAX typing forms cannot produce dtype/rank/key/domain static diagnostics. Numerical binding slots and topology sharing are now separate; runtime refusal tests own erased distinctions, and the installed-wheel consumer is included. Scientific review found no blocking contradiction and its positive-tail correlation/continuous-time cautions are explicit acceptance contracts above. No implementation or runtime result is implied by these reviews.

Binding slots use explicit source-term occurrence ordinals with term-ID provenance and checked canonical order. This does not relax the existing duplicate-term-ID refusal; the admitted unequal-value regression uses distinct term IDs with equal structural factor IDs. Resource maxima likewise remain distinct from the physical route bound used by semistochastic policy, preserving same-draw resource-only replay.

## Implementation disposition

Implementation remains on `native-projector-monte-carlo`; this plan is not kept
on `dev`. The worktree was updated to `dev` commit `85ad69ac5` without discarding
the ongoing implementation.

The canonical column-resource name is `QuantumColumnResourcePolicy`. The native
checkpoint owner now exposes one internal manifest-construction invariant, and
`lifecycle.create` accepts explicit native archive limits while preserving its
default policy and persistence format. Rich result publication retains canonical
static estimator interpretation/policies in addition to numerical leaves and
includes that interpretation in result identity.

Scientific review additionally tightened dependent-group correlation diagnostics,
nonzero guided-element underflow refusal, and bounded decoded-coordinate worksets.
All real/complex arithmetic has explicit precision at the mixed-kind boundaries.
Sparse column preparation and correlated-ratio analysis were decomposed into
preparation/binding/qualification phases; their ordinary helpers satisfy the
touched cyclomatic limit of 15 and measured NLOC limit of 120. The unchanged,
source-faithful host IPS ordering carries its numerical rationale.

Executed evidence is recorded in `benchmarks/projector_monte_carlo.json` and
`benchmarks/projector_monte_carlo_qualification.json`. Operational or finite-control
qualification does not promote the scientific candidate profiles to released
sign-cure, finite-population unbiasedness, or generic convergence claims.
