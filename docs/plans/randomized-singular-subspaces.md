# Native randomized singular subspaces: end-to-end implementation plan

## 0. Execution contract and scope

This is a design and execution plan, not an implementation or a performance claim. Repository source, relevant tests, native owners, and language-server references were inspected. No numerical code, tests, benchmarks, build, formatter, or linter was executed while preparing this plan.

The external reference is [pcax](https://github.com/alonfnt/pcax), inspected at commit [`894ff8905f62001352ee42d462805a5366338ba5`](https://github.com/alonfnt/pcax/commit/894ff8905f62001352ee42d462805a5366338ba5). Its useful mechanism is randomized range compression, not another PCA API. Mathematical references are [Halko, Martinsson, and Tropp](https://arxiv.org/abs/0909.4061) and [Halko, Martinsson, Shkolnisky, and Tygert](https://arxiv.org/abs/1007.5510). The latter describes block-Krylov/out-of-core techniques that the small reference implementation does not implement.

### Required deliverable

1. A native, bounded, leading randomized SVD method with original-operator residuals, range-error evidence, honest rank uncertainty, explicit random identity, and preparation/refresh/execution resource accounting.
2. Genuine first-order stopped, singular-value, basis, and invariant-projector derivative behavior. Exact dense results and finite randomized algorithms have distinct derivative meanings.
3. One shared native numerical owner used by the existing weighted ML subspace kernel and its consumers: PCA, TruncatedSVD, array POD, incremental PCA, fixed-query operator POD, ICA whitening, and fixed-subspace Onsager preparation.
4. End-to-end migration of changed result/model contracts, exact-rank consumers, sklearn conversion, ROM adaptation, tests, documentation, generated public/capability data, and a phase-separated scaling campaign.

### Explicit non-goals

- No dependency on pcax and no second fit/transform/recover facade.
- No absolute explained-variance API or change to current weighted/physical energy normalization.
- No block-Krylov method, restarted partial SVD, out-of-core data reader, automatic method choice, adaptive rank, automatic widening/retry, or hidden dense fallback.
- No LU normalization option. Start with QR; an additional normalization algorithm requires separate equivalence, derivative, and benchmark evidence.
- No change to PhysicalPODPlan's method-of-snapshots algorithm, its rank-resolution policy, or tensor-network truncation algorithms.
- No claim of scalable randomized support for arbitrary custom Hilbert pairings without a native admitted coordinate isometry. Dense arbitrary-pairing behavior remains supported.
- No higher-order derivative guarantee and no fit-through-optimizer-update derivative guarantee. First-order JVP/VJP of the explicitly admitted direct fit/merge surfaces is the qualification target.
- No unrelated decomposition/ICA/CCA/PLS cleanup, API alias removal, or broad test-suite restructuring.

### Implementation location

For a subsequent `//code` request, create a fresh worktree from the then-current branch under `/Users/lgleyzer/PHYDRA/phydra-labs/.worktrees/`, with a neutral branch/worktree name such as `native-singular-subspaces`. Copy the authoritative ignored `AGENTS.md` into that worktree before implementation. All package, test, benchmark, documentation, and generated-data changes remain there. A `//code--` request instead explicitly authorizes the current worktree. Do not discard concurrent changes or assume the source snapshots below are unchanged at execution time.

## 1. Baseline facts and invariants

### Native SVD

- `phydrax/linalg/_svd.py` currently implements one unbatched DenseSVD method. `count` and `which` select from a complete economy decomposition; they do not reduce decomposition dimensions.
- Its plan/preparation reserves/materializes the operator and square source/target pairing factors. Requesting a small count does not eliminate those allocations.
- Native differentiation currently admits only `none` and `singular-values`; exported vectors are stopped.
- Result `numerical_rank` currently denotes rank determined from the complete spectrum. That scalar cannot honestly represent a partial randomized spectrum.
- Native original-operator forward/adjoint residuals and metric orthogonality are valuable contracts and remain canonical.

### ML and consumers

- `ml/_numerics/_spectral.py` owns the shared weighted dense SVD. It uses sample weights normalized by their total mass, not sample-covariance division by sample_count - 1.
- `ml/decomposition/_subspace.py` owns masked weighted centering, support, diagonal physical feature metrics, affine encoding/decoding, and recipe contracts.
- Its projector/basis modes currently change declarations/diagnostics, not the raw SVD computational graph. The minimum-gap diagnostic conflates internal retained gaps with the retained/discarded boundary.
- `nn/operator/training/_pod.py` uses the same shared kernel and owns fixed query geometry, quadrature/masks, channel layout, and PODBasis construction.
- IncrementalPCA synthesizes paired pseudo samples from its previous retained covariance. Stopping previous public modes and singular values without replacing that covariance derivative would sever earlier-chunk sensitivities.
- Additional direct shared-kernel callers are real FastICA whitening in `_latent.py` and FixedSubspaceOnsagerModel.fit in `_onsager.py`. They are not FactorAnalysis or a separate PCA reconstruction class.
- sklearn conversion directly constructs SubspaceModel. ROM adaptation reads its physical components, support, metric, and offset.

### Existing reusable owners

- `linalg/eigen/_spectral_derivatives.py` implements genuine isolated-cluster projector differentiation with cross-membership denominators. Its current full-basis/dense-projector interfaces must be extended, not copied into ML.
- Native operators already provide coordinate block forward/adjoint actions and distinguish fused execution from column-equivalent work.
- `_spaces.py` already classifies coordinate-diagonal pairings and extracts their weights without materializing square Gram matrices, including supported composite spaces.
- `_sampling/_addressing.py` owns SampleAddress and derive_key. `_randomized_preconditioning.py` already has correct real/circular-complex Gaussian probe generation.
- `benchmarks/_runtime.py` owns synchronized timing, lowering/compilation separation, compiler evidence, logical resident-byte accounting, and runtime/build identity.

### Scientific invariants

Preserve existing dense forward semantics within independently specified numerical tolerances: weighted means, zero-extension of masked centered values, positive physical support, normalized sample energy, case ordering, complex conjugation, affine offsets, phase convention, fixed-query geometry, and ROM support refusal. Changes to derivative semantics and rank evidence are intentional clean cutovers and must be documented explicitly.

Scientific source/target identity comes from declared vector spaces and owning ML/query schemas, not equal dimensions or display strings. Anonymous array coordinates may remain explicitly anonymous; never identify sample and feature spaces merely because their sizes coincide.

## 2. Frozen public and numerical decisions

### 2.1 Methods, policy, and entry points

Keep the public native owner `phydrax.linalg.svd` and existing lifecycle names `plan_svd`, `prepare_svd`, `refresh_svd`, and `svd`.

Add a final StrictModule method `RandomizedSVD` with:

- `oversampling: int = 8`;
- `power_iterations: int = 2`;
- `probe_refresh: ProbeRefresh = "reuse"`, using one canonical selector declaration shared with randomized preparation utilities.

DenseSVD remains the default. Method dispatch is explicit and exhaustive. RandomizedSVD accepts `which="largest"` only; reject `smallest` at planning. Validate integer kind and value before assignment; booleans are not capacities. Do not silently convert fractional capacities with int(...).

Add an owning `SVDApproximationPolicy` for fixed audit probe count and total failure probability, with explicit numerical allowance semantics. Defaults: 8 independent audit columns and total failure probability 1e-6. It is part of canonical symbolic plan identity. Deterministic dense-access certification does not consume a random audit or claim a probabilistic failure probability.

Keep SVDTolerancePolicy as the owner of triplet residual and orthogonality thresholds. Do not secretly relax them when a randomized method is selected. Public examples must supply scientifically appropriate thresholds when exact-level defaults would reject an approximation.

Extend preparation and direct execution with keyword-only `key: PRNGKey | None = None`:

- Planning requires no key and draws nothing.
- Randomized preparation/direct execution requires an explicit scalar typed JAX key.
- Dense preparation rejects an extraneous randomized key instead of silently consuming it.
- Prepared execution rejects a replacement key; its numerical experiment is already bound.
- Refresh preserves the root key. A new root key starts a new preparation, not a disguised coefficient refresh.
- Reject legacy uint32[2] keys through the existing typing validator.

The ML recipes expose native `method`, `tolerance`, `resources`, and `failure` options alongside their existing component count and derivative target; these are the existing native types, not a parallel ML solver-policy hierarchy. A shared internal adapter constructs the native count/target/differentiation policy once. Key comes through fit_batch. Exact recipes remain key-free. Operator POD gains the same native options and an explicit fit key. ML defaults to status mode to preserve invalid-fit reporting; an explicitly selected error route raises. Incremental PCA retains dense merge decomposition in this cutover: its covariance derivative must be fixed, but a new randomized streaming API is not necessary for this deliverable.

### 2.2 Fixed randomized algorithm

Let C denote the operator in Euclidean orthonormal coordinates, m its target dimension, n its source dimension, k the requested count, and r = min(m, n).

Set ell = min(k + oversampling, r) once in the plan. Record requested and effective oversampling. The initial scan carry and every QR block have fixed admitted shapes. Full-size ell is allowed only when its declared costs fit; it is an explicitly full-size compressed algorithm, not a fallback.

Always use a source-domain sketch; do not transpose automatically for wide matrices:

1. Draw Omega with shape (n, ell).
2. Q = reduced_QR(C Omega).
3. Repeat a fixed number of times: Z = reduced_QR(C* Q), then Q = reduced_QR(C Z).
4. B = (C* Q)* with shape (ell, n).
5. Economy SVD only B; lift its left vectors through Q and retain the requested leading triplets.

Use native block actions. Long power refinement is a fixed lax.scan. There are no host scalar conversions, numerical Python loops over array axes, or runtime-created jit wrappers.

QR is computed at initialization and after both directions of every power step. No explicit normal matrix, inverse, jitter, clipping of the input spectrum, redraw after failure, or LU option.

For complex arrays, all transposes are Hermitian. Draw circular CN(0,I) probes from independent real/imaginary normals divided by sqrt(2), with explicit real/complex dtypes. Retained singular values and norm evidence have the corresponding real dtype.

Rank-deficient QR may return a finite orthonormal completion for a primal computation. Report rank/margin evidence; never label a completion direction as resolved sampled signal. Differentiated execution requires the admitted full-column-rank/conditioning margin at every QR. Expected rank loss caused by oversampling beyond signal rank still rejects algorithmic differentiation. Only the caller's explicit smaller fixed width changes that condition; there is no automatic repair.

### 2.3 Pairings and materialization

For positive diagonal source/target metric weights, let D_s and D_t be their square roots:

- C x = D_t A(D_s⁻¹ x).
- C* y = D_s A*(D_t⁻¹ y), where A* is the native Hilbert adjoint.

Reuse the existing diagonal-pairing classification and weight extraction. Randomized preparation supports those admitted Euclidean/diagonal spaces and their already-supported composite forms. Refuse arbitrary non-diagonal custom pairings before operator work. Dense SVD retains its general SPD-pairing route under explicit materialization budgets.

Also add a compact Euclidean/diagonal preparation branch to DenseSVD. ML migration must not replace its ordinary dense array SVD with unnecessary m²/n² identity factors. The general dense pairing branch remains available; do not change its scientific result or refusal ordering accidentally.

ML zero physical weights are represented by support masks and safe feature scaling, not by a singular native Hilbert pairing. The adapter supplies a Euclidean transformed operator and restores physical coordinates itself.

### 2.4 Random identity and refresh

Use SampleAddress/derive_key with separate static semantic roles for sketch and audit, tied to the symbolic plan identity and explicit case identity/index.

- Reuse mode addresses the sketch at numerical version zero.
- Redraw mode addresses it at the current numerical version.
- Both modes rebuild numerical Q/B when coefficients change.
- Audit always uses the current numerical version and a distinct role. Reusing audit probes after audit-dependent coefficient changes invalidates their independence.
- Repeated execution of one unchanged preparation reuses the same certificate; it is not a new confidence event.

Store root key words/implementation and derived role/version provenance as dynamic numerical identity where required. Do not host-hash traced key contents or bake changing keys into callable identity.

For a lifetime failure budget delta, assign version v the bound delta_v = delta / ((v + 1)(v + 2)); its infinite sum is at most delta. Compute the schedule without integer-product overflow and with a conservative representable allowance. Refuse numerical-version overflow rather than silently reusing a folded address.

The probability statement assumes audit draws are independent of current operator/sketch construction. No adaptive audit-driven tuning/retry is implemented. Record this assumption; deterministic replay is not proof that an adversarial input chosen from the audit key satisfies independence.

## 3. Approximation, rank, status, and resources

### 3.1 Range certificate and ordering

Triplet residuals alone cannot certify leading modes: exact triplets from a missed lower invariant subspace can have zero residual.

Certify the omitted range E = (I - Q Q*) C independently of the construction sketch. Let beta be the full descending compressed spectrum, and eta an upper bound for the spectral norm of E.

For structurally resident DenseLinearOperator data with supported diagonal metrics, compute a deterministic Frobenius residual upper bound in fixed column tiles. Accumulate the residual directly, with stable sum-of-squares and a declared numerical allowance. Do not subtract nearly equal total/captured energies to establish a small residual certificate, and do not allocate a second full transformed matrix. This is bounded processing of resident data, not an out-of-core API.

For a genuinely nonmaterializing operator, draw s independent unnormalized Gaussian audit vectors and compute z = max ||E g_j||:

- Real N(0,I): eta = sqrt(2/pi) delta_v^(-1/s) z.
- Circular complex CN(0,I): eta = z / sqrt(-log1p(-delta_v^(1/s))).

These are small-ball bounds, not heuristic standard errors. Normalized probes, Rademacher probes, and construction-probe reuse do not satisfy these constants. The theorem concerns the declared mathematical residual action; report floating-point allowances and any provider action-error uncertainty separately. Never label an opaque provider's unknown action error as an unconditional continuum certificate.

Report represented singular-value intervals beta_i <= sigma_i(C) <= beta_i + eta and omitted-spectrum upper bound eta. For k < ell, the discarded leading-tail upper bound is beta_(k+1) + eta; for k = ell, it is eta. An uncertainty-adjusted positive beta_k minus that tail upper bound certifies strict leading-cluster separation. Full-spectrum coverage can establish primal ordering without a positive internal gap.

Keep three separate facts:

1. Approximation/triplet/orthogonality numerical acceptance.
2. Strict leading-subspace ordering/separation certification.
3. Requested derivative admission.

A finite approximate result does not automatically establish the second or third. Add explicit leading-certification evidence and an owning policy requirement for consumers that demand a strictly certified partial leading cluster; those consumers cannot infer it from SUCCESS or finite values. Default randomized PCA/POD requests that claim a leading subspace must require this evidence, while native status-mode consumers can inspect an uncertified approximation without treating it as a certified leading result.

Do not require a small total rank-k reconstruction residual merely to admit low-rank fitting: even an exact truncation can have a large scientifically legitimate tail. Range/triplet tolerances and reported ordering evidence define approximation quality; tail energy remains evidence, not an invented universal admissibility threshold.

### 3.2 Two different reconstruction energies

Native low-rank factor reconstruction is Q B_k. ML's actual affine encoder/decoder uses C V_k V_k*.

- Native factor spectral residual upper bound: sqrt(eta² + beta_(k+1)²), with the missing compressed tail interpreted as zero when k = ell.
- ML component captured energy: ||C v_i||² measured using original-operator forward actions.
- ML retained energy: sum of those measured energies divided by its existing known normalized total.
- ML residual energy: the actual right-projection residual, not the factor residual or sum of compressed beta².

Preserve exact dense energy values/normalization within numerical tolerances. For randomized fits, document that per-component projection energies are measured energies, not exact covariance eigenvalues. Stop per-component diagnostic derivatives in projector mode; retained repeated clusters need invariant covariance derivatives, not individually differentiable labels.

A shell without an owning known total Frobenius energy has no total/capture fraction. Use explicit optional evidence with an availability/kind declaration, not NaN or zero standing for an unknown total. No new stochastic total-energy estimator is added.

### 3.3 Canonical rank evidence and clean cutover

Replace native diagnostics/result `numerical_rank` with one `SVDRankEvidence` contract:

- lower/upper global numerical-rank bounds;
- full-spectrum versus partial coverage;
- deterministic exact-rank availability;
- threshold lower/upper bounds;
- certificate kind and confidence/failure probability where applicable.

The threshold is the existing RankPolicy absolute + relative * largest-singular-value convention. Partial evidence must account for uncertainty in both unobserved modes and the largest value used to define the threshold.

For beta_1 <= sigma_1 <= beta_1 + eta, define tau_low = absolute + relative * beta_1 and tau_high = absolute + relative * (beta_1 + eta). A safe lower bound counts beta_i > tau_high; a safe upper bound counts beta_i + eta > tau_low, plus unresolved ambient modes when eta > tau_low, clipped to r. Include numerical allowances in the conservative direction.

Dense full-spectrum results retain exact numerical-rank availability even if count is small. Coincident probabilistic bounds are confidence-qualified, not promoted to deterministic exact rank. Core-local rank may be named as local evidence, never substituted for global numerical rank.

An owning exact-rank validator returns the equal bound only after checking successful deterministic exact evidence. Migrate exact-rank consumers to it. This is a real validation invariant, not a forwarding alias. Remove the obsolete native scalar property/field, not just deprecate it. Domain report fields named numerical_rank remain when their scientific contract is unchanged.

`RankPolicy.require_full_rank` must not pass just because every compressed singular value is positive. A randomized partial request requiring deterministic exact global rank is refused at planning unless full coverage and an admitted deterministic proof are available; confidence-qualified rank bounds remain explicitly different evidence.

### 3.4 Status and derivative failure

Retain separate primal and derivative status evidence and an aggregate requested-operation status. A usable primal can coexist with an unsupported requested derivative. Error mode raises the established native failure-policy error; status mode preserves fixed output shapes and exposes failure.

Add explicit statuses/evidence for uncertified requested leading ordering and uncertified required global rank. Preserve established dense nonfinite/rank/residual/derivative failure ordering unless a documented scientific contract change requires otherwise. Do not turn a stopped derivative into a supported gradient flag or suppress failed native status in ML.

#### Preparation failures and ML precedence

Preparation numerical failure is an explicit part of the native lifecycle, not an unconditional exception hidden before result assembly. Add a dynamic preparation status/evidence to each method-specific prepared state. Wrong object kinds, dimensions, selector values, incompatible declared semantics, missing keys, and inadmissible resources remain contract errors before numerical execution. Nonfinite operator data or failed numerical pairing-factor evidence obey the selected numerical FailurePolicy.

In status mode, a failed preparation and solve retain the same method-specific PyTree/array shapes under eager execution and jit. Do not choose a different Python state class on a traced validity condition. Skip unavailable QR/SVD/solve work with a shape-preserving lax.cond and return explicitly unavailable numerical arrays/evidence under the failed status; NaN payloads are never presented as usable results, rank has unavailable/uncertain coverage, and no plausible zero solution or successful decomposition replaces failure. In error mode, attach the native JIT-safe error to returned status/numerical roots so it cannot be an unconsumed check.

The shared ML adapter computes its owning input admission before decomposition:

- Sample/feature masks and zero sample weights exclude their nonfinite data by the existing declared zero-extension semantics.
- Preserve owner-specific status precedence, not a universal weight status. The direct shared kernel reports active nonfinite samples/nonfinite weights as ML_NONFINITE, finite negative weights as ML_INFEASIBLE, and zero total weight as ML_INSUFFICIENT_DATA. PCA/POD and operator-POD wrappers retain their existing override to ML_INFEASIBLE when their owning weight validity or metric/support check fails, including nonfinite, negative, or zero-total weights. An invalid wrapper metric can therefore take precedence over a spectral nonfinite cause; retain both causes in evidence without changing the public status.
- An invalid ML request does not run a sanitized-successful native fit. Return an explicitly rejected ML result with native numerical evidence unavailable, fixed-shape unusable fit arrays, and the original admission cause.
- An admitted ML request uses native status mode unless the user explicitly requested error mode. A subsequent native preparation/solve failure maps to ML failure and preserves its native cause/evidence, never overwritten by a success inferred from finite coefficients.

This intentionally repairs native status-mode preparation, while preserving ML error/status behavior for valid versus invalid requests. Document the changed native numerical-preparation refusal boundary. Tests must cover active NaNs, nonfinite/negative/zero-total weights at both the direct shared kernel and PCA/operator wrappers, masked and zero-weight NaNs, invalid/zero metric support, and native-only preparation failure in eager and jitted execution.

### 3.5 Resource admission

Plan before work, using native action-cost/resident-buffer utilities. Separate:

- original operator resident bytes;
- added preparation retained bytes;
- preparation/refresh workspace;
- solve/output workspace;
- forward/adjoint column-equivalent actions;
- actual fused block call counts;
- deterministic dense residual-scan arithmetic;
- exact versus inexact provider scratch estimates.

With q power iterations, preparation uses (q + 1) ell forward and (q + 1) ell adjoint column-equivalent actions; shell auditing adds s forward actions. Execution adds k forward and k adjoint original-triplet actions. Account extra derivative diagnostics and audit blocks explicitly rather than hide them in a single matvec count.

Retain only Q, B, admitted O(m+n) metric scales, key/version/provenance, and scalar/small evidence in randomized preparation. Do not retain Omega, all power iterates, the transformed operator, or square projectors. Fixed ell/probe widths are working-set capacities admitted against resources. Tile deterministic residual accumulation with lax.scan/fori_loop so tiles are reduced rather than concatenated into a full residual.

Target added storage/workspace scaling is O((m+n)ell + ell² + (m+n)s), plus declared operator scratch and requested outputs. Full-size ell honestly admits matrix-size storage. Dense complete decompositions retain their own explicitly bounded cost. Unknown opaque provider workspace stays inexact; no exact resource certificate is manufactured.

## 4. Differentiation and fitted-model invariants

### 4.1 Exact dense derivatives

Extend the existing eigen derivative owner with compact array-level cross-block and divided-difference operations. Existing eigen APIs remain adapters and keep their behavior. Do not send a rectangular SVD through dense eigen preparation, form A* A, or materialize a Hermitian dilation just to differentiate it.

For C = U Sigma V*, H = U* dC V, retained directions S and discarded thin directions D:

- Right cross response: (Sigma_D H_DS + H_SD* Sigma_S) / (sigma_S² - sigma_D²).
- Left cross response: (H_DS Sigma_S + Sigma_D H_SD*) / (sigma_S² - sigma_D²).
- Include omitted source/target null complements with thin-block subtraction, not full complement matrices.

Only retained/discarded denominators occur in projector response. Repeated values inside either cluster are legal. A positive retained/discarded gap and appropriate nonzero/null-mode conditions are required. A full ambient-space projector is identity with zero spectral tangent; tall/wide left/right cases differ and must be handled explicitly.

Basis derivatives compute only retained-column responses, without discarded/discarded denominators. Require positive individually isolated retained values and measured pivot magnitude/top1-minus-top2 margins. In complex arithmetic account for diagonal phase connection and differentiate the fixed locally unique canonical pivot. Choose the right-vector pivot and rotate both left/right vectors by the same unit phase so A V = U Sigma remains true.

Compute singular-value derivatives through the whitened operator and coordinate transforms. Include both moving metric terms; the current target-inner-product-only custom derivative is insufficient for moving source/target metrics. For metric G_s/G_t, the first-order correction includes (sigma/2)(u* dG_t u - v* dG_s v). Use native triangular solves or diagonal actions, never an explicit inverse.

### 4.2 Finite randomized derivatives

The random sketch is fixed and stopped. Differentiate all admitted QR/power steps and B = Q* C. Attach the compact selected projector response to B's rectangular decomposition, not an exact-original-C projector rule.

- Right projector response is the compressed B response.
- Left response includes Q P_left(B) Q* and the derivative of Q.
- Individual randomized singular-value derivatives are derivatives of beta(B), not Re(u* dA v) presented as an exact-original singular derivative.
- Admission uses QR rank/conditioning, compressed boundary separation, and basis pivot/individual-isolation conditions as appropriate.
- Scientific approximation/order certification is separate. A derivative of a finite algorithm is not a proof of the derivative of the exact original leading subspace.

First-order finite differences reuse the same key, addresses, numerical version, and pass count. Report an algorithmic/unrolled derivative route using the existing DerivativeRoute vocabulary; do not invent another differentiation framework.

### 4.3 Mode behavior

| Mode | Fit-dependent basis/value outputs | Projector/covariance response | Affine mean | Independent prediction-input/model-parameter gradients |
| --- | --- | --- | --- | --- |
| none | Stopped | Stopped/absent | Stopped | Ordinary prediction derivatives remain |
| singular-values (native) | Only positive individually isolated values differentiate | Stopped | Not owned by native SVD | Ordinary consumer derivatives remain |
| projector | Raw representatives and individual values stopped | Genuine invariant response; internal repeats allowed | Differentiable on admitted fixed support | Ordinary prediction derivatives remain |
| basis | Canonical retained triplets/values differentiate with actual margins | Derived consistently | Differentiable on admitted fixed support | Ordinary prediction derivatives remain |

Stop at fit construction, not in prediction methods. A stopped fit does not prohibit optimizing or differentiating an independently supplied current model.

### 4.4 Compact response without a stale projector cache

A fitted ML model must not retain the original samples, complete thin decomposition, operator closure, or a feature-by-feature projector. Attach derivatives while those objects are still owned by the native solve, then retain compact output arrays only.

Keep one authoritative weighted basis. Derive physical-coordinate components from it and the metric rather than retaining separately mutable physical and weighted basis copies.

For projector-mode fitting, construct private tangent-only correction carriers:

- T_W = W_response - stop_gradient(W_response), identically zero in the primal;
- T_K = K_response - stop_gradient(K_response), identically zero in the primal.

W_response has the retained frame as its primal and only discarded/null horizontal response in its tangent. K_response has retained squared singular values on its primal diagonal and the complete selected-selected covariance perturbation in its tangent, including off-diagonal terms at repeated retained values.

Projector/covariance actions consume the current authoritative basis plus these zero-primal corrections. The model does not cache an independently mutable projector basis. Corrections are explicitly fixed/nontrainable role leaves; they are derivative carriers, not fake scientific zeros or fallbacks. Their primal-zero invariant and role partition are owned and tested.

Consequences:

- Primal project equals inverse_transform(transform(x)) for the same current basis, including after an ordinary independent basis parameter update.
- Direct fit-project derivatives receive the invariant horizontal/covariance response even though raw fit-basis outputs are stopped.
- Independent parameter-lane gradients use the current basis with correction lanes held fixed; never suppress parameter gradients inside transform/inverse/project.
- Prediction-only/sklearn model construction has no fit correction/provenance and uses its current basis directly.
- Direct fit-response claims apply before basis reparameterization/mutation. Fit-through-optimizer/basis-update differentiation is explicitly unsupported; do not claim the original spectral certificate applies to a learned replacement basis. Owner-controlled prediction-only reconstruction clears fit-response provenance/corrections. Arbitrary external PyTree edits are not an admitted route for obtaining a new certified fit.

Include correction/core arrays in resource and PyTree-role accounting. Do not put dynamic arrays in static fields. Preserve canonical scientific identity payloads for basis/offset/geometry independently of zero-primal response bookkeeping.

### 4.5 Operation-specific derivative admission

The callable SubspaceModel is an encoder. In projector mode it must not advertise full fit-basis gradients for encoding just because project() supports an invariant derivative.

Extend the existing FitResult/DerivativeContract admission owner with an immutable optional tuple of operation-specific contracts/gates, keyed by canonical operation identifiers. This is evidence metadata, not a numerical dispatch registry. Existing families without method-specific behavior retain their existing default callable contract.

The canonical return owner is `phydrax/_differentiation.py::DerivativeAdmission`, not a new fit wrapper. Its current `supported` boolean, string `status`, levels, route, conditions, and reasons remain static **declaration eligibility**; they do not resolve numerical conditions. Add paired optional dynamic `runtime_valid` and `runtime_status` evidence arrays, with None meaning no runtime condition resolution was supplied. Validate the pair, dtype, request-surface alignment, and case shape before assignment. Add an owning with-runtime-evidence construction that preserves the static declaration and returns the same canonical admission type. Never host-convert a traced gate, put it in a static field, or turn missing evidence into True.

For subspaces, explicitly describe transform, inverse_transform, project, and projector. Default admission refers to the model's callable/transform surface. Projection admission is requested for project/projector. A projector-mode encoder/decoder does not admit a full fit-basis derivative; its deliberately retained affine-mean path is documented separately. Preserve fit-feature/fit-weight gates separately from independent input/model-parameter surfaces. `FitResult.derivative_admission(..., operation=...)` attaches only the runtime gates relevant to the requested surfaces; a failed fit gate cannot revoke independent model/input eligibility.

`FitResult.require_derivative` first preserves the existing static ValueError rejection, then guards resolved numerical gates with an existing JIT-safe error primitive and returns the checked runtime_valid as part of the admission. An unresolved unrelated family's existing conditional declaration remains conditional, not newly proved or silently rejected. The actual native response/error route also enforces requested differentiation admission; a user ignoring admission metadata does not obtain a falsely supported custom derivative.

`phydrax/ml/_fit.py` currently ignores the return of require_derivative. Change that boundary to consume the checked runtime gate by attaching it to the returned FitResult/model numerical roots when derivative_request is supplied. An unused assertion/callback result is not enforcement. OperatorPODFit uses the same canonical declaration-plus-runtime-evidence representation for projection, coefficient transforms, and PODBasis decoding. Schema binding and incremental wrapping preserve it. Unknown operations fail explicitly; generic code never probes arbitrary diagnostic attributes.

### 4.6 Incremental covariance bridge

Keep the existing rank-truncated merge algorithm, chunk order, paired plus/minus primal normalization, support masks, mean, and total mass. It is not exact streaming PCA and is not changed into a new randomized streaming method.

Carry the compact retained covariance response. Build the existing pseudo rows from a stopped square-root factor F = V Sigma, attaching an owning smooth factor tangent instead of using stopped public modes/values as derivative inputs.

For retained covariance differential dC_k, let H_SS = V* dC_k V. Set X_ij = H_SS_ij / (sigma_i + sigma_j), and dF = V X + (I-P) dC_k V Sigma⁻¹. This satisfies d(F F*) = dC_k, tolerates repeated positive retained values, and requires only feature-by-rank actions. No dense covariance is retained.

Differentiate prior offsets and total weights too. Every merge's boundary/rank admission reaches the final result. The finite-difference oracle is the same rank-truncated merge algorithm, not full batch PCA, which is a different scientific computation.

## 5. File-by-file implementation map

New files below are proposed canonical owners, not existing files assumed to have an implementation.

### P1 — Native contracts, metric preparation, and exact derivatives

| File | Required change |
| --- | --- |
| `phydrax/linalg/_svd_contracts.py` (new) | Own method/policy/state/result/status/evidence declarations extracted from the current monolith, including RandomizedSVD, approximation policy, canonical rank evidence, separate primal/derivative evidence, and stage costs. Validate before assignment; use final StrictModule classes, precise annotations, canonical selector parsing, and explicit source/target/mode dimension roles. |
| `phydrax/linalg/_svd.py` | Retain public lifecycle orchestration, exhaustive method admission/dispatch, plan identity/refresh checks, shared original-operator diagnostics and result assembly. Remove extracted obsolete definitions/import paths after migrating their internal callers. Decompose by planning, preparation, execution, admission, and result phases rather than line-count fragments. |
| `phydrax/linalg/_svd_dense.py` (new) | Own exact dense preparation/decomposition: compact Euclidean/diagonal scales, general SPD pairing-factor branch, full thin spectrum, coordinate restoration, and exact rank evidence. No wrapper around an unrelated dense linear-solve backend. |
| `phydrax/linalg/_singular_subspaces.py` (new) | Own compact singular-projector/covariance actions, retained response carriers, coordinate transforms, shared row-phase canonicalization/pivot evidence, and local covariance-factor tangent attachment. ML/cross/latent callers import the canonical phase capability directly; no ML centering/masking here and no full-projector cache. |
| `phydrax/_differentiation.py` | Extend the canonical DerivativeAdmission with optional paired runtime evidence and owning validation/construction. Preserve static declaration semantics for all existing users; numerical fit gates do not become static booleans or invented proofs. |
| `phydrax/linalg/eigen/_spectral_derivatives.py` | Reuse/extend cross-membership and divided-difference numerics for compact rectangular responses and selected covariance density. Existing eigen adapters preserve their public output/error contracts. No normal-matrix materialization. |
| `phydrax/linalg/_spaces.py` | Reuse diagonal classification/weights; add a shared scale preparation helper only if both dense/randomized owners use it and it owns positivity/support/dtype invariants. Do not change unrelated space identities or metric formats. |
| `phydrax/linalg/_pairings.py` | Preserve canonical pairing interfaces; no new generic scalable square-root abstraction. Change only if required to expose existing diagonal capability truthfully. |
| `phydrax/linalg/svd/__init__.py` | Explicitly export canonical new types/actions and lifecycle functions from their real owners. Remove obsolete scalar rank surface. No compatibility aliases. |
| `phydrax/linalg/__init__.py` | Preserve scoped svd facade and avoid collision with the unrelated linear-solve DenseSVD class. No second top-level spelling for the new method. |

Native derivative admission is implemented before ML begins consuming stopped vectors, preventing an intermediate silent loss of existing fit sensitivities.

**Import layering:** array-level eigen derivative helpers and existing spaces/operators are below `_singular_subspaces`; that compact action/response owner imports no SVD policy or runtime worker. `_svd_contracts` imports only those lower owners, not dense/randomized/evidence workers. `_svd_evidence` and `_svd_dense`/`_svd_randomized` consume contracts and lower owners. `_svd.py` orchestrates them; the public `svd` facade imports declarations from their canonical owners. This avoids contracts/runtime/eigen cycles without private compatibility re-exports.

### P2 — Randomized preparation and certification

| File | Required change |
| --- | --- |
| `phydrax/linalg/_svd_randomized.py` (new) | Fixed source-domain range refinement, QR margins, key-bound Q/B preparation/refresh, compressed SVD, finite-algorithm derivative graph, and method-specific live-set estimates. |
| `phydrax/linalg/_svd_evidence.py` (new) | Original-triplet/orthogonality assembly shared with dense route, tiled deterministic omitted-range measurement, independent real/complex small-ball audit, singular/tail/order bounds, and partial rank evidence. Helpers own distinct evidence invariants. |
| `phydrax/linalg/_randomized.py` (new, only shared real invariants) | Canonical ProbeRefresh selector and existing real/circular-complex Gaussian probe substrate extracted from randomized preconditioning. Preserve the existing Nyström seeded stream and identities exactly. No shared PSD or range algorithm abstraction. |
| `phydrax/linalg/_randomized_preconditioning.py` | Migrate only genuinely shared selector/probe imports. Nyström algorithm, seed/address behavior, builder fingerprints, and scientific evidence remain unchanged. |
| `phydrax/linalg/_operators.py`, `phydrax/linalg/_costs.py` | Consume existing fused block actions and action/resident-cost contracts. Extend only a missing generally useful cost/capability fact; do not add a local vmap substitute or pretend opaque scratch is exact. |
| `phydrax/_sampling/_addressing.py` | Reuse existing public address/derive operations unchanged. Add no SVD-specific RNG implementation here. |
| `phydrax/linalg/_rank.py` | Reuse threshold ownership. Keep full-spectrum numerical_rank_data consumers unchanged; partial-spectrum inference belongs to SVD evidence. |

### P3 — Shared weighted adapter and consumers

| File | Required change |
| --- | --- |
| `phydrax/ml/_numerics/_spectral.py` | Remove direct decomposition from _fit_one; adapt admitted normalized weighted arrays to native status-mode SVD, carry response/order/rank evidence, preserve ML admission/status precedence and actual projection-energy semantics. Invalid requests skip successful decomposition and return unavailable evidence as specified in section 3.4. Remove obsolete `_canonicalize_rows`; migrate every cross/latent reference directly to the owning shared phase capability, with no forwarding shim or second phase implementation. |
| `phydrax/ml/_numerics/__init__.py` | Export the canonical shared result/fit adapter only; update extracted owner imports without redundant re-exports. |
| `phydrax/ml/decomposition/_subspace.py` | Pass native method/tolerance/resources/key and differentiation target. Split cutoff, internal isolation, and pivot diagnostics. Use one authoritative weighted basis, derived physical components, compact correction actions, honest default encoding contracts, and metric-correct project/projector. Preserve fixed support, sample/feature masks, case axes, physical scaling, and affine restoration. |
| `phydrax/ml/_contracts.py` | Add optional operation-specific DerivativeContract/gate metadata and operation-aware admission to existing FitResult. Return the canonical declaration-plus-runtime DerivativeAdmission, guard requested resolved gates, preserve unrelated families' conditional declarations, and carry metadata through bind_schemas/frozen extraction without attribute probing. |
| `phydrax/ml/_fit.py` | Consume the checked runtime admission root when derivative_request is supplied; attach its error gate to returned result/model numerical roots so jitted consumers cannot discard it accidentally. Preserve the existing schema/fit entry-point semantics. |
| `phydrax/ml/decomposition/_incremental.py` | Carry the retained covariance response and native factor tangent through existing pseudo-row merges. Preserve numerical primal normalization, total mass, masks, chunk order, and dense merge selection. Replace unsupported fit-basis derivative declarations with per-operation evidence. |
| `phydrax/ml/decomposition/_latent.py` | FastICA whitening explicitly requests exact dense basis-mode behavior and carries isolation/pivot refusal into its existing unrolled fit evidence. Migrate row-phase calls/imports to the canonical shared owner. Preserve FastICA iteration/RNG algorithm and other latent methods. |
| `phydrax/ml/decomposition/_cross.py` | Migrate row-phase imports/calls directly to the canonical owner after removal of the ML helper. Preserve numerical results and operation order; do not alter CCA/PLS algorithms. |
| `phydrax/nn/models/wrappers/_onsager.py` | FixedSubspaceOnsagerModel.fit explicitly requests dense none mode; migrate exact rank evidence through the owning validator into the unchanged physical projection report. Preserve fixed basis/dynamics semantics. |
| `phydrax/nn/operator/training/_pod.py` | Pass native options/key, use canonical derivative target parser, propagate evidence, add physical project/projector operations, keep basis decoder semantics distinct from projection fit sensitivities, and preserve query fingerprints/quadrature/channel layout. |
| `phydrax/nn/operator/architectures/conditioning/_deeponet.py` | Preserve PODBasis evaluation/query checks and independent parameter/input gradients. Only update fit provenance/contract ownership if needed; no blanket stop inside evaluate. |
| `phydrax/ml/interop/_sklearn.py` | Update both PCA and TruncatedSVD constructors to prediction-only response representation. Preserve imported transform/inverse values and existing whitened-PCA refusal; no fake fit source, random key, or native success evidence. |
| `phydrax/rom/_basis.py` | Read derived physical components from current authoritative basis; preserve complete positive physical-support refusal, scientific space/geometry/measure IDs, offset, and artifact construction. Do not imply projector-mode fit gradients support differentiating a basis-sensitive reduced model. |
| `phydrax/ml/decomposition/_physical_pod.py` | Intentionally unchanged algorithm/API. Its arbitrary-pairing snapshot-Gram policy is separate; documentation distinguishes it from the new array/native routes. |

Batching uses declared case identity and bounded execution. Replace an unbounded all-case decomposition vmap with admitted fixed-size case blocks using existing execution support or lax.map with a fixed batch size. Preserve output order and case RNG addressing independent of block capacity. When using eqx.filter_vmap for a homogeneous native PyTree block, set in_axes explicitly; static metadata is shared and not mapped/device data.

### P4 — Exact-rank consumer migration

Language-server references identify the following removed native scalar-rank callers:

| File | Migration invariant |
| --- | --- |
| `phydrax/applications/radiation_biophysics/_qualification.py` | Require successful deterministic exact evidence before converting rank to the existing host qualification report. |
| `phydrax/applications/incompressible_flow/_control.py` | Exact evidence gates full-rank acceptance and the existing response/condition report. |
| `phydrax/applications/polymer_liquids/_prism.py` | Full-rank claims use global exact evidence, not selected/core rank. |
| `phydrax/applications/battery/_identifiability.py` | Validate exact global rank before existing weak-mode/condition decisions. |
| `phydrax/discretization/particle/_rigid_constraint_dynamics.py` | Preserve rank-valid/constraint-domain failure behavior after evidence validation. |
| `phydrax/solver/_mac_immersed_boundary.py` | Preserve marker-rank report and refusal; no guessed rank in a field solve. |
| `tests/unit/test_linalg_svd.py` | Assert exact coverage/bounds and scientific statuses instead of removed property spelling. |

Other default dense callers that consume only singular values/status do not need gratuitous API rewrites. Re-run LSP references at implementation time, including diagnostics/model constructor references, then supplement dynamic/interchange coverage with targeted searches. Do not assume today's reference set covers concurrent additions.

## 6. Test contracts and migration ratchet

Tests below cover consumer-visible behavior, not imports, source text, copies, or internal dispatch. Use small bounded independent NumPy/analytical references. Mark relevant dtype/rank-promotion boundaries strict_jax. Parametrize genuine shape/dtype/metric matrices with diagnostic IDs. Preserve nonfinite-mask and failure-status assertions.

### Native exact and response tests

- `tests/unit/test_linalg_svd.py`: retain dense lifecycle, refresh/stale state, largest/smallest ordering, zero/rank-deficient output, explicit full-rank refusal, pairing orthogonality, closure-converted operator gradients, and materialization/resource refusal. Migrate exact-rank assertions; do not weaken them.
- `tests/unit/test_linalg_singular_subspaces.py` (new): independent projector JVP/VJP values, internally repeated retained/discarded clusters, boundary crossing, tall/wide ambient null complements, per-side full-space identity, positive/zero selected modes, complex phase, source/target moving metrics, and isolated-basis pivot ties/margins.
- `tests/unit/test_linalg_spectral_subspaces.py` and `tests/unit/test_linalg_self_adjoint_spectrum.py`: existing eigen projector/density behavior remains numerically and diagnostically unchanged after common-kernel reuse.
- Verify A V = U Sigma after shared phase canonicalization and differentiate that relation under admitted perturbations. Do not compare raw gauges at repeated spectra.
- Use nonconstant asymmetric probe contractions for gradient checks. A sum of squared orthonormal basis entries is not a meaningful derivative oracle.

### Randomized tests

- `tests/unit/test_linalg_randomized_svd.py` (new): tall/wide including m=8,n=20,k=5 with width clipped to both dimensions; float32/64 and complex64/128; supported diagonal/composite pairings; non-diagonal custom pairing refusal only in randomized mode.
- A shell whose materialize capability is false exercises real block actions and never requires dense storage. Compare observed numerical result/evidence to a known small operator, not mock forwarding counts alone.
- Fixed-key direct/prepared/refresh reproduction; reuse/redraw sketch semantics; audit independence/versioning and summable confidence; legacy/missing/wrong-shape key refusal.
- Zero and rank-deficient sketches have finite fixed-shape primal output and honest rank evidence. Differentiated QR rank loss is explicitly rejected; no hidden repair.
- An adversarial captured lower invariant subspace has near-zero triplet residual but fails required leading-order certification.
- Direct deterministic omitted-range residual matches an independent reference. Independently derive real/complex bound constants using rank-one analytical cases; do not gate CI on flaky empirical coverage frequencies.
- Hidden unobserved modes keep global rank uncertainty. Full-rank policy cannot be satisfied by a positive compressed spectrum alone.
- Distinguish native factor reconstruction residual from actual right-projector residual on a deliberately inexact sketch.
- Fixed-sketch finite differences exercise dependence through QR and nonzero power steps. Do not compare finite randomized derivatives to exact dense derivatives as their oracle.
- Error/status modes, requested-leading evidence, derivative support, opaque action scratch, and resource refusal remain visible.

### ML and downstream tests

- `tests/unit/ml/test_numerics.py`: preserve weighted reconstruction/energy/nonfinite masking. Delete the affected constant-frame-norm finite-gradient assertion; replace it with a genuinely nontrivial projector/covariance action derivative comparison. Leave unrelated weighted least-squares scenarios unchanged.
- `tests/unit/ml/decomposition/test_subspaces.py`: exact and randomized PCA/TruncatedSVD/POD projection behavior, actual captured energy, sample/feature masks, complex phase, zero physical support, case ordering, admitted fit-feature/fit-weight projector derivatives, mode-specific mean/basis behavior, operation-aware admission, current-model parameter derivatives, and projection/transform-inverse primal equality after independent basis updates. Delete wording-pinning contract assertion; do not repin new text.
- `tests/unit/ml/decomposition/test_incremental.py` (new coherent owner): two/three-chunk final-projector derivatives with respect to an earlier chunk and earlier weight, repeated positive prior retained cluster, per-merge boundary failures, offset/mass response, support preservation, and the same rank-truncated merge finite-difference oracle. Keep the existing batch-projector comparison where its setup genuinely preserves the same subspace.
- `tests/unit/ml/decomposition/test_latent_factorizations.py`: ICA whitening/fit derivative admission and existing numerical unrolled behavior; no broad latent-method rewrite.
- `tests/unit/nn/test_operator_pod.py`: physical project action/derivative, centered mean, quadrature, channels, fixed-query geometry refusal, basis-mode decoder fit derivatives, projector-mode stopped fit-basis decoder route, and independent decoder/branch/input derivatives.
- `tests/unit/nn/test_port_hamiltonian.py`: fixed-subspace Onsager preparation remains a valid fixed scientific model; add a meaningful fitted-snapshot case if coverage is absent.
- `tests/unit/ml/interop/test_sklearn.py`: imported PCA/TruncatedSVD transform, inverse, and current-basis projection agree with the provider while no fit provenance is manufactured. Missing sklearn skips only at its canonical boundary.
- `tests/integration/test_coupled_rom.py` and `tests/unit/fidelity/test_rom.py`: run only affected existing basis/offset/support consumers; preserve physical IDs and accepted/refused intrusive support semantics.
- `tests/unit/ml/test_contracts.py`: operation-specific fit admission, dynamic refusal, preservation through bind_schemas, and unchanged default callable admission for a non-subspace family. `tests/unit/ml/test_fitted_model_roles.py`: actual parameter partition/update behavior, fixed correction/core lanes, and current-basis projection/encoding equality; do not merely assert role metadata.
- `tests/unit/test_differentiation.py`: paired runtime-evidence validation, unresolved evidence versus static eligibility, preservation of the existing static declaration contract, and canonical eager/JIT numerical gate propagation. In `tests/unit/ml/test_contracts.py`, toggle an operation's fit gate under one compiled callable, observe runtime unsupported evidence/refusal, preserve independent input/parameter admission, and test a fit(..., derivative_request=...) consumer that uses only the returned model.
- `tests/typing/cases/singular_subspaces.py` (new): truthful native methods/evidence/model operation types, typed key acceptance and negative legacy/wrong-kind boundaries, result projection shapes, and operation-specific admission. Pair reachable misuse with runtime refusal. Execute through existing `tests/typing/test_typecheck.py`.

### Rank-consumer selection

After exact-rank access migration, select the existing scenario items that exercise the changed decisions in:

- `tests/unit/applications/radiation_biophysics/test_initial_lesion_qualification.py` and/or `test_radiation_calibration.py`;
- `tests/unit/applications/test_incompressible_flow_control.py`;
- `tests/unit/applications/test_polymer_liquids.py`;
- `tests/unit/applications/battery/test_identifiability.py`;
- `tests/unit/discretization/test_rigid_constraints.py`;
- `tests/unit/solver/test_mac_immersed_boundary.py` and the affected pipeline integration scenario.

Choose exact collected node IDs from the existing tests at execution time rather than run unrelated domain suites. Preserve domain report types/messages and scientific failure precedence.

### Coverage map and deliberate faults

Maintain an old-to-new contract map for the affected shared-kernel/native tests: dense fit, energy, masks, phase, rank, lifecycle, derivatives, and each failure remain owned by independently collectable scenarios. Do not reduce assertions/counts as evidence of coverage. Capture a touched-module coverage ratchet if the resulting changes constitute a material suite refactor.

Deliberate local fault checks must catch at least: omitted null-complement tangent, detached earlier incremental covariance, reused construction probes for audit, wrong complex conjugation, compressed-energy substitution for projection energy, and core-rank-as-global-rank. Revert each deliberate fault before the single final verification pass; they are adequacy checks, not committed mutation infrastructure.

## 7. Runtime smoke and scaling benchmarks

### Driver and records

Add `tools/singular_subspace_benchmarks.py` using `benchmarks/_runtime.py`, and canonical `benchmarks/singular_subspaces.json` evidence from the actual run. Keep bounded argument validation, American identifiers, deterministic row ordering, and existing build/environment identity. Add no schema/generation version fields. Do not modify the shared benchmark harness merely to accommodate this campaign.

One actual driver invocation must smoke the changed paths as well as record performance:

1. Native nonmaterializing rectangular operator preparation, solve, refresh, compact projector action, and a witnessed refusal/status path.
2. Weighted/masked PCA and physical array POD with asymmetric probe projection and admitted first-order derivative.
3. Fixed-query operator POD encoder/decoder and physical projection.
4. A repeated-cluster exact projector and an earlier-chunk incremental derivative.
5. A deliberately poor fixed approximation with observable certification failure, not a silent fallback.

These are actual numerical workflows, not source inspection or mocked API calls. Observe outputs, residuals, statuses, captured energy, and derivative comparison. Tests alone are not smoke proof.

### Phase separation

Report separately:

- host symbolic planning and admission;
- cold preparation;
- refresh with unchanged structure/changing coefficients;
- lowering;
- compilation;
- first synchronized execution;
- warmed prepared execution;
- warmed projection/encoding/decoding actions;
- differentiated action execution;
- compiler argument/output/temporary/generated-code bytes;
- logical original-operator, added-preparation, result, fitted-model, and response-carrier bytes;
- action counts, provider scratch exactness, and measured device memory where supported.

Do not blend factor preparation with repeated projection speedups or claim compiler estimates are measured peak memory. A prepared decomposition is reused for downstream actions. If repeated prepared svd recomputes the small core decomposition by the public lifecycle contract, report that cost rather than advertise factor reuse it does not implement; ML stores fitted compact outputs for repeated projection without refitting.

### Capacity/spectrum matrix

Use independent m/n sweeps, not only one small example. Include tall/wide families, k and oversampling sweeps, q=0/1/2, real/complex dtypes, diagonal metrics, resident dense and genuine block-action shell routes, and exact dense comparison where admitted.

Controlling examples: m/n in bounded geometric sequences such as 64/256/1024, k in 4/16/32 where valid, oversampling 0/4/8, and fixed audit capacity. Record complete route/resource status for refused rows instead of omitting them. Include fast decay, slow decay, near-cutoff clusters, exact low rank, and rank-deficient sketches.

Acceptance is not an arbitrary universal speedup number. Evidence must demonstrate bounded added-storage scaling with fixed ell/probe capacity, absence of square metric/projector allocations in randomized execution, preservation of quality/evidence, and the actual crossover against dense execution. If certification is conservative or a full-size/slow-spectrum row is slower, report it. Do not tune away unfavorable cases or switch methods invisibly.

Before/after measure changed symbol complexity: cyclomatic and cognitive complexity, NLOC, physical lines, parameters. Use the repository's existing tooling where available or direct configured Ruff C901 plus an existing installed cognitive-complexity tool; do not add a second repository-wide checker. New ordinary helpers target <=15 cyclomatic, <=20 cognitive, and <=120 NLOC. Decompose severe orchestrators by the real lifecycle/evidence phases above while preserving reduction/error ordering.

## 8. Documentation, qualification, and generated surfaces

| File | Required update |
| --- | --- |
| `docs/api/linalg.md` | Exact/randomized method table, supported pairings/capabilities, key/refresh identity, stage costs, uncertainty/rank contract, leading certificate, finite algorithm versus mathematical derivative, compact actions, and failure/refusal examples. |
| `docs/api/ml/unsupervised.md` | Native method/tolerance/resources options; weighted/masked normalization; actual projection energy; exact/algorithmic modes; default encoder versus explicitly admitted projector gradients; incremental covariance response. |
| `docs/guides/ml.md` | One weighted PCA and physical POD example using the existing API; explicit randomized key/tolerance and evidence checks; no second PCA vocabulary. |
| `docs/appendix/ml_differentiability.md` | Correct operation-level claims and first-order boundaries; internal cluster repetitions, cutoff/pivot margins, QR admission, moving metrics, stopped-fit versus independent model parameters, and unsupported fit-through-basis-update route. |
| `docs/api/nn/architectures.md` | PODBasis decoder versus physical projector gradient semantics; retained fixed span/geometry restrictions and new fit options. |
| `docs/guides_reduced_order_modeling.md` | When snapshot Gram, exact orthonormal-coordinate SVD, and randomized array POD are appropriate; preserve resolution-floor and physical-support guidance. |
| `CHANGELOG.md` | Consumer-visible new randomized capability, genuine derivative behavior, operation admission, rank-evidence clean cutover, supported/refused metrics, and unchanged normalization. |
| `phydrax/linalg/_svd_qualification.py` (new) | Own exact unreleased singular-subspace profiles and qualification observations using existing qualification contracts: method, representation, pairing, dtype, derivative route, certification kind, and resource envelope. Use actual campaign/scenario evidence; no signed release claim. |
| `phydrax/qualification/_builtin_catalog.py` | Register the native profile producer through its existing lazy provider list. Do not append a profile to the all-core portfolio or broaden its dense-LU tuple; existing all-core observation coverage remains unchanged. |
| `docs/data/public_api.json` | Regenerate with `tools/generate_public_api_manifest.py` after public export/type changes. |
| `docs/data/capabilities.json`, `docs/api/capabilities.md`, `docs/data/application_portfolios.json`, `docs/data/capability_closure.json`, `docs/data/source_absorption.json`, `docs/api/capability_closure.md` | Regenerate with `tools/generate_capability_inventory.py` after exact profile changes; retain only canonical generator changes induced by declarations/current concurrent state. Unchanged source-ledger data stays unchanged. |
| `NOTICE`, `LICENSES/PCAX-MIT.txt` | Update only if upstream source is copied/adapted. Prefer independent native implementation of the cited mathematics. Preserve required MIT notice for any copied substantial portion; cite the reference design/papers regardless. |

Update affected examples/qualification callers discovered by LSP/reference searches in the same cutover. Do not hand-edit generated manifests, create release claims from tests, or introduce versioned artifacts/compatibility aliases.

## 9. Implementation and verification order

1. Create the authorized worktree, copy the authoritative context, inspect current differences since the last merge into dev, and refresh public/reference maps. Preserve concurrent changes. Record phase-separated dense baseline evidence and reproduce the relevant current gradient-contract gaps with small consumer workflows before editing.
2. Implement native contract extraction, compact diagonal dense preparation, canonical rank evidence, compact exact derivatives, and operation-aware admission. Update broken consumer contracts in the same phase; no user-facing stopped-vector cutover without replacement projector/covariance behavior.
3. Implement fixed randomized preparation, independent audit/certification, finite-algorithm derivatives, resources, and RNG refresh identity.
4. Migrate the shared weighted adapter, existing subspace/operator models, incremental covariance bridge, ICA/Onsager preparation, sklearn imports, ROM consumers, and exact-rank decisions.
5. Update coherent behavioral tests, typing fixtures, documentation, profiles, and explicit public exports. Remove superseded numerical paths, stale comments/imports/fields, and misleading incidental tests; introduce no compatibility path.
6. Run the actual smoke/capacity campaign, observe its values/status/evidence, and retain canonical measured records. Remove throwaway scripts after proof.
7. Run one conservative final affected-test set with `-n auto`, except a benchmark/device-isolated provider scenario that genuinely requires dedicated execution. No full suite merely because linalg is important; expand selection only when current changes alter global collection/configuration/shared fixtures or another genuinely cross-cutting contract. Never rerun a user-reported failure merely to confirm it.
8. Run configured formatting/lint once after integration, `python tools/check_typing.py check`, `python tools/audit_selectors.py`, affected typing fixtures, and installed-wheel typing when public annotations change. Regenerate/check public and capability inventories using their existing commands. Required first-party typing/annotation diagnostics are zero.
9. Reconcile benchmark/resource/derivative claims with actual evidence, confirm every affected caller/document uses the canonical representation, and report exercised verification plus explicit supported/refused capability limits. Do not call unfinished kernels or consumer migrations an MVP/foundation.

Illustrative final focused command, expanded only by the exact affected consumer node IDs found above:

```text
python -m pytest -n auto \
  tests/unit/test_linalg_svd.py \
  tests/unit/test_linalg_singular_subspaces.py \
  tests/unit/test_linalg_randomized_svd.py \
  tests/unit/test_linalg_spectral_subspaces.py \
  tests/unit/test_linalg_self_adjoint_spectrum.py \
  tests/unit/test_differentiation.py \
  tests/unit/ml/test_contracts.py \
  tests/unit/ml/test_fitted_model_roles.py \
  tests/unit/ml/test_numerics.py \
  tests/unit/ml/decomposition/test_subspaces.py \
  tests/unit/ml/decomposition/test_incremental.py \
  tests/unit/ml/decomposition/test_latent_factorizations.py \
  tests/unit/nn/test_operator_pod.py \
  tests/unit/ml/interop/test_sklearn.py
```

This is a future verification command, not a run performed while planning. New test filenames are proposed owners. Add existing Onsager/ROM/operation-admission and exact-rank scenario node IDs only when their actual changed consumer contracts require them; retain provider isolation and canonical optional-dependency skips.

## 10. Completion gates and risk decisions

The implementation is complete only when all of these hold:

- Exact dense default behavior and physical/masked normalization are preserved, except the explicitly documented derivative/rank-contract cutovers.
- A genuinely nonmaterializing randomized operator path runs without operator/square-metric/projector materialization and refuses inadmissible resources/capabilities before work.
- Full/key-bound preparation, refresh, execution, fixed-shape failures, and stage evidence are implemented, not scaffolded.
- Approximation, strict leading certification, rank uncertainty, and derivative admission reach the consumer as distinct facts.
- Exact repeated-cluster projector JVP/VJP and finite randomized algorithm JVP/VJP have value-based proof, including ambient null complements and moving metrics.
- Raw basis/decoder gradients are not mislabeled as invariant projector gradients; none stops fit outputs, not independent model evaluation.
- Earlier incremental chunks retain their admitted covariance derivative; no stopped-basis sensitivity loss is hidden.
- Fitted models retain only compact current basis/response/core state; projector primal remains synchronized with encoding/decoding after independent basis updates, with no false fit-through-update certificate.
- All named additional consumers and removed exact-rank accesses are migrated; obsolete native numerical/rank paths are absent and domain report meanings are preserved.
- Documentation, typing, profiles, generated surfaces, actual smoke results, and phase-separated scaling evidence agree with the shipped capabilities.

Main tradeoffs are explicit: QR prioritizes a tractable derivative/correctness contract over an unearned LU speed claim; fixed source-domain sketching preserves covariance-factor invariance over automatic wide-matrix orientation; custom non-diagonal randomized pairings are refused rather than secretly densified; independent certificates can be conservative on slow spectra; deficient-QR primal computation can be useful while its finite-algorithm derivative is refused. Dense remains the explicit exact alternative, not an automatic rescue path.
