# Native runtime boundary contracts: end-to-end implementation plan

## 0. Status, scope, and binding decisions

This plan was written read-only and then implemented; section 13 records the implemented outcome and the final per-guard disposition map. Source changes made concurrently by others remain theirs.

Development baseline reconciled with merged [PR #383](https://github.com/phydra-labs/phydrax/pull/383), `dev` and current HEAD at `85ad69ac5`. This plan builds on that current canonical meshfree/compiler-identity representation; it does not preserve or restore pre-PR representations.

The implementation goal is to remove independently maintained nominal argument guards where annotations can own the same contract. It is not to eliminate every `isinstance`, implement every Python typing feature, or move scientific validation into a generic checker.

### Binding user decision: preserve existing callable IDs

The user chose **Preserve existing callable IDs** over allowing code-derived IDs to change.

`phydrax/_identity.py` identifies a stateless importable function from its code, defaults, and referenced globals. Removing an argument guard changes that code. PR #383 also binds exact compiler programs, custom-JVP source/transformations, closure captures, and imported helper bindings. `functools.wraps` preserves neither identity path: an ordinary checking wrapper is opaque to `callable_payload`, while a wrapper captured by compiler metadata can change its bound source payload or be refused. Even unwrapping cannot recover the former bytecode after a source edit.

Therefore:

1. Preserve existing source-addressed scientific function bodies, defaults, referenced identity inputs, and module bindings. This includes private module-level functions when they can be content-addressed, and nested code contained in a protected function.
2. Do not decorate an existing identity-sensitive free function just to leave its original guard executing underneath a second check.
3. Concentrate guard removal on constructors and audited methods only after proving that both field-based identity and compiler-bound source/callback identity are unchanged. Explicit semantic/numeric IDs at one consumer do not exempt a callable from another consumer's exact compiler binding.
4. Keep `_identity.py` and its source fingerprint semantics unchanged. Do not add a wrapper-unwrapping registry, a historical code-payload cache, compatibility IDs, aliases, or relaxed opacity rules.
5. The new `checked` decorator follows the current callable-identity policies. A new wrapped free-function callback is opaque to `callable_payload` and requires explicit IDs there. Exact compiler metadata has no general explicit-ID escape hatch for arbitrary checker closures: bound checker plans, locks, and wrapper state can be refused as unowned metadata. Do not promise that a checked callback can participate in every native compiler-bound workflow, or that explicit IDs solve that separate admission contract.
6. Preserve existing scientific object, plan, RNG, revision, provenance, persistence, and resource payloads. The new public API manifest necessarily gains a symbol and changes its own generated manifest identity; that is not a scientific callable-ID migration.

This decision intentionally constrains free-function cleanup. It must not be obscured by a claimed repository-wide zero-guard target.

### Final attachment decision

Use **one explicit `phydrax.typing.checked` function decorator**, including on audited custom `__init__` methods. Do not add blanket metaclass enforcement, an import hook, a class input-opt-in flag, or a separate constructor decorator.

This refines the earlier speculative automatic-metaclass proposal. It avoids resolving every annotation in every class, prevents new checking from preempting generated-constructor converters, keeps the existing Strict lifecycle intact, and uses the same public mechanism for methods and constructors. After a boundary is decorated, its supported annotated arguments are checked automatically on invocation.

Generated constructors without existing argument guards are not an excuse to broaden this change into global runtime enforcement. Their opted-in stored-field contracts remain enforced by the existing structural checker.

### Other fixed decisions

- Native `_typing_plan.py` remains the contract owner. No beartype/typeguard dependency and no jaxtyping package annotations or import hook.
- The decorator validates arguments, not returns. Return semantics, callback results, failure/status, and transformed-module validation stay with existing owners.
- No implicit conversion, selector canonicalization, device transfer, scalar synchronization, or parameter rewriting.
- Plain numeric/scalar and conversion annotations do not acquire stricter meanings. In particular, do not redefine plain `int` as exact Python int or plain `float` as one global numeric-tower policy.
- `Literal`/Enum selector inputs continue to be parsed by their current owners. Preserve accepted NumPy scalar spellings and canonical stored values.
- A fresh dimension Scope belongs to one checked invocation, not to a class, thread, cache, nested object tree, or return value.
- Static-only coverage is documented and audited. Unsupported placement of Phydrax vocabulary remains a hard error, not a silent shape-check omission.

## 1. Baseline evidence and exhaustive file appendix

### Observed ownership

| Owner | Current responsibility |
|---|---|
| `phydrax/_typing_forms.py` | Nominal dimensions and tensor/metadata aliases; static base types; canonical vocabulary |
| `phydrax/_typing_plan.py` | Structural grammar, Scope/rollback, contract checking, categorized violations, cached class plans, field/tree validation |
| `phydrax/typing.py` | Public parse/conversion/validation boundaries |
| `phydrax/_strict.py` | Abstract/final rules, Equinox construction integration, post-construction field validation, freeze |
| `phydrax/_validation.py` | Identifier and scalar acceptance/normalization/value validation |
| `phydrax/_identity.py` | Scientific callable/object identity plus exact compiler programs, native callback/source/transform bindings, and fail-closed metadata ownership |
| `phydrax/_trainable.py` | Hidden callable state; trainable-global inspection versus compiler `include_imported=True` binding |
| `phydrax/_model/_structure.py` | Constructor-bypassing reconstruction and complete-tree validation |
| `tools/generate_public_api_manifest.py` | Publicness from explicit ordered exports, including lazy facades |
| `tools/audit_contract_candidates.py` | Read-only structural contract candidate analysis |

### Static inventory snapshot

The companion file is [runtime-contract-files.tsv](runtime-contract-files.tsv). It enumerates every package Python file in a canonical path order, with a source SHA-256, guard/class counts, candidate symbols, and prescribed work categories.

At the latest static snapshot used here:

- 5,999 package Python files;
- 32,232 syntactic `isinstance` calls;
- 11,270 exact, single-raise parameter-annotation guard matches;
- 11,058 non-primitive annotation-spelling candidates;
- 6,543 method-owned candidate sites across 2,432 files;
- 4,515 source-function-protected candidate sites across 1,801 files.

These are syntactic candidates, **not** 6,543 promised safe deletions. An unresolved alias, protocol, conditional narrowing, scalar policy, or identity-bearing enclosing function can disqualify a site. The method/source split is conservative lexical ownership, not a substitute for actual identity and API analysis.

The tree changed during planning. The appendix was refreshed from current source; each row's counts and checksum come from the same read. Refresh against the implementation worktree before making changes. Do not restore source to match the appendix.

### How to use every appendix row

| Work category | Required file-specific action |
|---|---|
| `audit-effective-constructor-input-contracts` | Resolve actual Strict/StrictModule/custom/dataclass ownership; inspect constructor inputs, converters, inheritance, methods, and identity implications. Do not assume every listed class is a StrictModule. |
| `migrate-proven-method-and-constructor-guards` | Evaluate every named method candidate; attach `checked` and remove only fully owned guards, or record an explicit retained disposition. |
| `check-compiler-callback-identity-before-guard-migration` | Prove the method/constructor and any imported helper or wrapper cannot alter a current bound compiler/source payload. Otherwise retain the site as compiler-source-protected. |
| `retain-native-compiler-source-ownership` | Preserve the exact callback/program/rule ownership listed in compiler_review_targets; field checks and nominal attachment cannot substitute for source/capture admission. |
| `retain-source-addressed-function-bodies-and-guards` | Protect the listed function and enclosing source-addressed code from body/default/binding changes. No checking wrapper is added just for cleanup. |
| `retain-semantic-dynamic-conversion-and-identity-guards` | Classify remaining guards; preserve dispatch, scalar/conversion, generic-container, provider-output, scientific, reconstruction, and security ownership. |
| `verify-export-or-annotation-dependency-no-edit-unless-affected` | Inspect only if export resolution, runtime annotation imports, or a migrated caller reaches this file. No gratuitous modification. |

Files with both method and protected-function work need surgical edits to methods only. A source-addressed factory containing a local class is protected as a whole; its local methods are not automatically eligible.

The implementation must produce a final old-to-new contract disposition map keyed by canonical owner symbol and parameter, with original guard location, new owner, actual callers, verification scenario, and retained reason when applicable. Keep this in the existing candidate-audit report, not in scientific persistence formats or the public API manifest. No generation suffixes or schema-version fields.

### Reconciliation with PR #383

The core typing/Strict ownership has not changed. The attachment decision remains explicit checked constructors/methods, with no import hook, blanket enforcement, or new identity adapter. Required plan updates are in migration eligibility and downstream proof:

1. **Method identity needs a second gate.** `_CompilerPayload.bound_function` hashes code, defaults, closure contents, and imported helper bindings. Methods carried by native bound callbacks or static PyTree metadata are not inherently safe just because the containing module's ordinary scientific ID is field-based.
2. **No added JAX equations is insufficient.** A checker may vanish from the primal Jaxpr but remain in custom-JVP source, native callback closure, transformation metadata, or static tree ownership. Compare actual canonical execution metadata and its admission behavior, not only numerical output/equation counts.
3. **Keep guarded nonlinear transforms unchanged.** The PR's batching, JVP, and partial-evaluation rules preserve lane-local rejection, unmapped primal outputs, accepted-branch residuals, and failed-state sensitivity. The signature wrapper must not force both branches, evaluate a derivative to infer types, turn an unmapped primal into mapped output, or replace a failed derivative with success.
4. **Current meshfree ownership is mandatory.** The canonical package is `phydrax.discretization.meshfree/`; the old `meshfree.py` and `_local_polynomial.py` are deleted. Use `PointCloudPlan(stencil=LocalStencilPolicy(...), neighbors=...)`, not superseded wrappers or dense Poisson/dissipative paths.
5. **Preserve new evidence consumers.** Rank/conditioning/support/sign/resource refusal, source-program/numeric-revision admission, complete epoch history transfer and rollback, failed forward/adjoint refusal, and `FixedStepResult.evidence` are scientific contracts, not removable type boilerplate.
6. **Regenerate from the post-PR catalog.** The public API/capability data already includes meshfree. Adding `typing.checked` must preserve those entries. Candidate status is not a scientific release qualification; do not promote meshfree capability profiles as part of this typing change.

The refreshed appendix already contains the 22 current meshfree package Python files. Its conservative exact-guard scan finds seven method-owned and four protected free-function candidates there; all 107 syntactic isinstance sites still need their actual ownership classification, not blanket replacement.

### Concrete post-PR source and callback protection map

| Current owner/symbol | Exact protection requirement |
|---|---|
| `_identity._CompilerPayload.bound_function` / bound_callable | Code, defaults, closure cells, imported helper bindings and native partial recursion; bound MethodType is not automatically a transparent FunctionType callback. |
| `_identity._CompilerPayload.parameter_payload` / program | Explicit native compiler/callback/StrictModule/partition ownership, source programs and capture declarations; preserve foreign primitive, effect, placement, symbolic-extent and unowned metadata refusal. |
| `_identity._CompilerPayload.rule_owner` / rule_selector | Only JAX-owned memoization and recognized flatten/static-argument/batch/lift transformations; no arbitrary __wrapped__ traversal. |
| `_identity.execution_metadata_payload` | Current canonical metadata DAG and separately declared dynamic versus static captures; do not erase static arrays or add debug/object-address identity. |
| `_trainable._global_reads` | Keep default training-state inspection distinct from compiler include_imported=True recursion; imported helpers are real compiler bindings, not name-only aliases. |
| `PreparedSurfacePointCloud.prepared_id` | Nested source_field/ambient_metric -> jax.make_jaxpr -> execution_metadata_payload -> SemanticProvenance/NumericRevision -> reconstruction source owner and MeshfreeComponent admission. |
| `RegularLevelSetManifold.constraint` / ambient_metric; CompiledGeometry boundary/projection kernel path | Forward the actual source callable/program unchanged; coincident point/operator arrays cannot establish source identity. |
| `MovingSurfacePlan.geometry_refresh` / reaction | Static callback fields may capture operator arrays; a checked constructor must retain the original callback objects and their owner IDs. |
| `FixedStepCouplingParticipant.method` / bind / observe / amounts / estimate_error | Preserve native method and callback fields; MeshfreeBulkSurfaceMethod.participant's nested bindings remain source-owned. |
| `_domain_cond._batch` / _jvp / _partial_eval / _transpose / domain_cond | Preserve guarded known/mapped outputs, symbolic-zero paths, accepted-branch residuals, lane-local effects and transposed branch execution. |

The appendix's compiler_review_targets column records concrete symbols or callback roles on 16 current owner files, including nonlinear `_types.py`, dynamics `_differential_algebraic.py` (`_ActiveDAESetupOperator`), solver `_dae_events.py`/`_implicit_stage.py`, and battery `_circuit_ecm.py`. These are source/callback review targets, not added guard-removal counts.

[INFERENCE] Independent nominal checks in MeshfreeOperator.__init__, MeshfreeAdvection.__init__, and PreparedHyperviscosity.__init__ remain plausible migrations if actual canonical fields and program admission/payload are unchanged. MovingSurfacePlan, MeshfreeComponent, and exchange constructors are mixed boundaries: validate a nominal argument without replacing or decorating callback values, and keep owner/reconstruction/program checks. No callback-bearing method is approved wholesale from its annotation.


## 2. Runtime semantics

### 2.1 Distinguish input and stored-state contracts

Constructor input annotations describe accepted inputs. Stored fields describe canonical state. A constructor accepting `ArrayLike` may convert once and store `Float64[NodeDim]`; the stored-field contract must not be applied to its raw input.

The existing field path remains:

1. Strict abstract refusal;
2. Equinox effective generated/custom/inherited `__init__`;
3. Equinox field converters;
4. every owning MRO `__check_init__`;
5. Phydrax opted-in field validation in declaration order;
6. ordinary Strict freeze; no extra instance flag on StrictModule.

A checked custom constructor inserts argument validation immediately at its own entry, after Python has bound the call and before its body assigns fields. It does not move field validation before converters or `__check_init__`.

Do not put checks in `tree_unflatten`, `tree_at`, or generic reconstruction hooks. `typing.validate` and `_model/_structure.py` remain the consumer/restore checks for transformed state.

### 2.2 Supported signature grammar

| Annotation category | Input behavior |
|---|---|
| Audited runtime nominal class | Check object kind; subclasses accepted according to the existing `isinstance` contract. Exact-type registry rules are not replaced. |
| Optional / closed union of supported nominal forms | Deterministic alternatives; `None` accepted only when declared. |
| `Callable[...]` | Check only `callable(value)`; no invocation, arity inspection, result checking, closure conversion, or signature probing. |
| Canonical Phydrax array, key, Size, Identifier, Identifiers forms | Reuse structural metadata checking when the signature actually promises canonical input. |
| Supported optional/union/fixed tuple of canonical forms | Reuse the canonical grammar, one Scope, and rollback. Native/nominal combinations must be explicitly representable before migration. |
| Plain scalar types (`int`, `float`, `complex`, `bool`, `str`, `bytes`) | Static-only in this layer; retain owning scalar/identifier validators and canonicalization. |
| `ArrayLike`, `ConvertibleToArray`, `SupportsArray`, `Like[...]` | Conversion/static-only; the body remains the single conversion owner. |
| `Literal[...]` and selector Enum annotations | Static-only at input-signature stage; keep owner `parse`. Existing stored-field selector checks remain unchanged. |
| General generic containers, TypeVar, Protocol, `type[T]`, arbitrary Annotated metadata, provider-specific typing | Static-only unless an already supported native contract applies. Existing container/provider/type-registration validation remains mandatory. |
| Phydrax vocabulary in an unsupported placement | Qualified `TypeError` during plan compilation. Never silently discard the vocabulary. |

Do not infer scientific axis identity from nominal dimensions. `Dim` binds extents; `phydrax.axes` still owns scientific identity.

For a union containing ordinary static-only alternatives, do not enforce only its nominal alternatives and reject values accepted by the skipped branch. Leave the ordinary mixed union static-only and report coverage. If native vocabulary is present in a combination that the owner cannot faithfully represent, fail compilation rather than claim enforcement.

The checker is deliberately not an exhaustive PEP type checker. The audit must identify supported checks and explicit owner coverage for each skipped material contract. A decorator with no effective checks is an audit finding, not a silent success path for a migrated boundary.

### 2.3 Binding, defaults, and errors

- Native Python binding errors must win over contract checks: missing/duplicate/unknown arguments and positional-only/keyword-only misuse remain binding `TypeError`s.
- Supported arguments are checked in signature declaration order, including omitted defaults. Within an argument, retain the existing native precedence of kind, shape, dtype, and dimension evidence. This is not a global 'all TypeErrors before all ValueErrors' pass.
- Wrong nominal kind, non-callability, backend, dtype, or key kind raises `TypeError`; wrong rank, extent, dimension binding, or minimum raises `ValueError`.
- Use qualified owner/parameter paths for diagnosis. Tests assert categories and meaningful diagnostic ownership, not incidental prose.
- The body is not entered on an input-contract failure. Numerical status/error behavior after a valid structural entry remains unchanged.
- Retain a guard when its conditional position expresses a narrower branch contract, or when moving it before conversion changes a required single-fault behavior.
- Multi-fault precedence can change when a previously late nominal guard becomes an entry guard. Enumerate intentional changes by target, document them, and do not label the migration as wholly exception-order-preserving.
- Variadic arguments obey their actual Python annotation meaning: `*args: T` describes each positional member and `**kwargs: T` each keyword value. Do not treat the annotation as a contract for the enclosing tuple/dict. Migrate only audited finite/native cases; dynamic keyword payloads retain their owner.
- Do not consume arbitrary iterables, inspect callback return signatures, execute converters, or copy argument containers.

### 2.4 Compilation, resolution, and caching

Compile a signature plan lazily once for a selected callable. Resolve only parameter annotations, not return annotations, against the defining module and actual defining class/MRO owner. Preserve runtime annotations and signatures read-only; do not rewrite Equinox's field annotations.

Why lazy: a method decorator runs before its class is bound, and forward references may be completed later in module import. Why not silently skip unresolved selected names: that would make declared boundary checking import-order-dependent.

Selected contract-bearing annotations must resolve. A checked target whose runtime names cannot resolve gets a qualified compilation error. A static-only hint with a deliberately unavailable provider also needs an explicit audit disposition; do not import every optional provider or move every TYPE_CHECKING import into module startup. Preserve genuinely cycle-dependent/manual boundaries when a safe selected plan cannot be formed.

Plan cache ownership belongs to the function/wrapper or weak callable keys. Cache only immutable plans, never an invocation Scope, bound user arguments, numerical arrays, or instances. Avoid name-based keys and weak-key values that strongly retain their own key. Dynamic same-name classes/functions must not reuse another object's plan.

Hot successful calls must not repeatedly call `get_type_hints`, resolve aliases, build decorators/JIT wrappers, or run `inspect.signature`. Do not make `inspect.Signature.bind` plus `BoundArguments`/fresh kwargs reconstruction the unconditional warmed path. Prepare parameter positions/defaults once; use native signature binding or a validated slot plan. Any generated binding shim must interpolate no user annotation/source text, preserve defaults by identity, and be justified by benchmark evidence. Start with boring local binding code, not a general code-generation framework.

Nominal/callable-only plans should not allocate a dimension Scope. Use the owning nominal/callability predicates directly; structural plans allocate one fresh Scope only when binding is needed. Avoid copying unchanged tuples or rebuilding kwargs merely to pass values through.

### 2.5 Decorators, descriptors, transformations

The public signature is conceptually `checked[**P, R](function: Callable[P, R], /) -> Callable[P, R]`, with truthful ParamSpec/return typing under the pinned ty.

Use the standard order, with `checked` closest to the function:

```python
@eqx.filter_jit
@pt.checked
def evaluate(...): ...

@classmethod
@pt.checked
def prepare(cls, ...): ...

@jax.custom_jvp
@pt.checked
def primal(...): ...
```

- Preserve signature/name/doc metadata and normal descriptor behavior.
- Exclude method receiver `self`/`cls` from argument contracts. Do not infer a receiver from a free-function parameter merely named `self`; selected target ownership must be known.
- Check custom-derivative primals, not tangent/cotangent rules. Preserve `.defjvp`/`.defvjp` APIs.
- The custom-derivative example illustrates placement for a new non-identity-protected boundary. Do not apply it to an existing compiler-bound primal/rule whose source declaration would change. JIT/AD compatibility and compiler-identity compatibility are separate acceptance gates.
- A checker inside a compiled callable runs during tracing; outside one it runs per host invocation. Do not claim unconditional zero JIT overhead.
- Checking adds no device runtime data, equations, numerical materialization, or host synchronization.
- Equinox method wrappers and unflattened module copies must not retain checker plans in instance fields or auxiliary metadata.
- Do not stack `checked` twice or also add a second automatic metaclass check. If an existing transform prevents a safe placement, preserve that boundary's owner guard and record the reason.
- Checking that a callback argument is callable must forward that exact object. Never replace geometry_refresh, reaction, bind, observe, constraint, metric, residual/trial-validity, or branch callback values with checked wrappers at the callsite.

## 3. File-by-file runtime implementation

### 3.1 `phydrax/_typing_plan.py` — extend the canonical owner

Add coherent private input-compilation/checking responsibilities, reusing the current Contract/Scope/Violation substrate:

1. Nominal and callable-presence contract variants with deterministic labels and no scientific value semantics.
2. `compile_input_form` for the limited input grammar, separate from `compile_form`'s stored-field language. The latter keeps ordinary annotations static-only.
3. Nominal/callability predicates shared by the complete dispatcher and nominal fast path. Do not duplicate these checks in the decorator module.
4. Reuse the existing array/key/size/identifier/union/tuple checker and Scope rollback. Extend central dispatch explicitly when an input variant is added; do not add an opaque dynamic registry.
5. Callable-parameter forward annotation resolution using the existing typing_extensions resolution approach and defining owner context. Keep field_annotation semantics unchanged.
6. An immutable input coverage record that distinguishes enforced, static-only/owner-owned, and refused placements. This is developer/audit metadata, not device or persistence data.

Do not turn adding nominal input contracts into nominal field enforcement. `ClassPlan`, `class_plan`, `validate_instance`, `validate_constructed`, and `validate_tree` retain declaration order, independent object scopes, canonicalize=False, exact categories, and current restoration semantics.

Measure materially changed symbols before/after with the repository's complexity tooling. Separate grammar compilation, annotation resolution, validation, and result/error construction by real invariants; keep ordinary new helpers within the style limits.

### 3.2 `phydrax/_typing_signature.py` — new private execution adapter

Create only the signature/binding/wrapper adapter, not another structural engine.

Responsibilities:

- Compile/capture an immutable ordered parameter plan lazily for `checked`.
- Resolve defining owner context for constructor/method annotations without importing public `phydrax.typing` or `_strict` into the compiler cycle.
- Bind/check supported arguments, forward the original values, and return the body's unchanged result.
- Preserve ParamSpec/runtime signature/descriptor metadata.
- Own plan cache lifetime and concurrency. Do not cache failures as a success/no-op plan.
- Reject accidental duplicate decoration rather than quietly adding repeated checks; use narrowly owned function metadata, never module instance metadata.
- Provide the audit/declaration layer a read-only way to compile coverage for selected targets without constructing instances or running product computations.

The public decorator implementation can live here and be exported only through `phydrax.typing`. Do not export private helpers through unrelated facades.

### 3.3 `phydrax/typing.py` — public API and explanation

Export `checked` explicitly in the existing `__all__` order. Update the module docstring to distinguish input signatures from stored-field contracts and list its deliberately static-only categories.

Leave the bodies and behavior of `parse`, `as_array`, `as_host_array`, and `validate` unchanged. Keep selector canonicalization, explicit one-time conversion, and transformed-tree validation. Do not add `phydrax.checked` or another spelling.

### 3.4 `phydrax/_strict.py` — intentional non-change

Keep metaclass construction/freeze code, abstract/final precedence, TYPE_CHECKING-hidden __call__, and AbstractVar normalization unchanged. Checked custom `__init__` is entered through the existing Equinox lifecycle; no new global input hook is needed.

Validate this non-change with constructor/lifecycle tests. If an integration defect requires touching `_strict.py`, stop the caller rollout, state the specific lifecycle invariant, revise this plan, and retain a focused regression before proceeding. Do not silently switch to blanket enforcement.

### 3.5 Core files intentionally unchanged

| File | Preserved contract |
|---|---|
| `phydrax/_typing_forms.py` | Vocabulary and static alias base types; no new exact-int/conversion aliases |
| `phydrax/_validation.py` | Real/Integral admission, bool exclusion, normalization, positivity/finiteness, canonical identifiers |
| `phydrax/_identity.py` | Current post-PR bytecode/global/default, exact compiler program/capture/custom-JVP binding, primitive/effect/placement ownership and opaque-callable refusal |
| `phydrax/_trainable.py` | Hidden numeric state and the distinct include_imported policies; no generic unwrapping exemption |
| `phydrax/_model/_structure.py` | Exact recipe inventory and once-complete restore validation |
| `phydrax/_model/_artifacts.py` | Exact registered-type artifact/architecture restoration |
| `phydrax/_array_archive.py` | Allocation preflight, backend/dtype/shape/resource/checksum/security admission |
| `pyproject.toml`, `uv.lock` | No new checker/provider dependency, language, or formatting convention |
| Installed Equinox source | Read-only reference; never patched |

Do not alter public free-function bodies in these files as incidental cleanup.

## 4. Complete caller migration

### 4.1 Build the actual target set

The appendix is the exhaustive static checklist. Join it with:

- public API paths from `tools/generate_public_api_manifest.py` and ordered facade exports;
- explicit documented member lists for methods, including intentionally public dunders;
- effective constructor inheritance and Equinox converter annotations;
- actual callsite/reference results from the language server;
- callback/provider/identity consumers and source-addressed enclosing-code ownership.
- current compiler-program/callback consumers, source custom-JVP declarations and imported helper bindings from `execution_metadata_payload`, not only `callable_payload`.

Use language-server references before changing exported signatures/imports. Do not infer publicness from filename underscores. No actual public input signature should be narrowed simply to make a checker compile.

Every non-primitive candidate receives one disposition:

1. **migrated-constructor**: checked constructor owns the same raw parameter contract;
2. **migrated-method**: checked method owns the same raw parameter contract;
3. **retained-source-identity / retained-compiler-source**: body, default/global/closure identity input, enclosing code, exact compiler callback source or transformation binding is protected;
4. **retained-canonicalization/scalar**: an owner admits/coerces values beyond the proposed kind check;
5. **retained-capability/semantic**: branch-specific scientific/status/identity contract;
6. **retained-provider/container**: output ABI, protocol, nested element, registry, or decoder contract;
7. **retained-resolution/decorator**: runtime import cycle, optional-provider name, or transform order cannot safely form the selected input plan;
8. **not-a-supported-boundary**: local predicate/helper semantics are not an entry-contract replacement.

No unresolved/unclassified candidate is acceptable at delivery. New concurrently added files receive the same analysis; never delete them or roll them back to the planning snapshot.

### 4.2 Per-target edit procedure

For each eligible method or constructor:

1. Read the full signature, relevant guard blocks, value/conversion phases, decorator stack, and consumers.
2. Establish the accepted domain, including subclasses, None, NumPy scalars, convertible inputs, and single-fault errors.
3. Establish every identity consumer: ordinary callable/field IDs, exact compiler program and custom-JVP source, static PyTree metadata, recursive callbacks and imported helper bindings. Protect surrounding source-addressed factories/nested code. If unchanged admission and payload cannot be proved without changing the identity owner, retain the guard as compiler-source-protected.
4. Make selected nominal annotations runtime-resolvable only where an owning import is cycle-safe and does not load an optional provider eagerly. Otherwise retain the guard with a resolution disposition.
5. Add `checked` from its canonical public owner. Generated/private framework internals must not import the public facade into a cycle.
6. Remove only complete, redundant guard blocks. Preserve branches that narrow after a selector, guard decoded/untyped data, or validate an object created by conversion.
7. Keep subsequent source/target compatibility, field identity, dtype/rank/value checks, result/status evidence, capacities, and failure rollback.
8. Update every caller whose annotation or actual input construction changes, including tools, examples, static fixtures, provider adapters, and benchmarks. Default: no caller change when accepted inputs are preserved.
9. Update its contract disposition and meaningful existing scenario ownership. Do not add one repetitive per-class wiring test.

Do not delete a private method's guard merely because ty sees its annotation. Either retain it at a dynamic trust boundary or give the method its own checked entry. Removing a downstream guard on the strength of a checked caller requires complete callgraph/ownership proof, including transformed state and provider calls.

### 4.3 Representative concrete files

The following files were inspected by the planning investigations. Apply the same ownership rules to every appendix file; this table is not a limited first-wave deliverable.

| File | Constructor/method changes | Must remain |
|---|---|---|
| `phydrax/linalg/_operators.py` | Audit Identity/Jacobian/Scaled/Transpose/Adjoint/DualTranspose constructors; replace direct operator/space/properties guards where input plans own them. Audit non-constructor methods individually. | Dense matrix conversion, batched-space restrictions, action dtype, properties/capabilities, closure conversion, eval_shape output validation, custom pairing behavior, NotImplemented operator protocols. Top-level adjoint/transpose helper bodies remain protected. |
| `phydrax/linalg/_problems.py` | Audit LinearSystem and other problem constructors' nominal operator/policy entry guards. | Square/source-target semantics, nullspace meaning, problem identity and numerical evidence. |
| `phydrax/linalg/_spaces.py` | Audit nominal pairing inputs only where runtime-resolvable and accepted kinds agree. | Sequence shape admission, exact/static integer policy, dtype conversion, pairing semantics and space ID. |
| `phydrax/linalg/_plans.py` | Audit LinearSolvePlan constructor only. | Source-addressed `plan` and method-validation free functions; GMRES/FGMRES recycling, differentiation, precision, sparse/structured, property, resource, and solve capability gates. |
| `phydrax/linalg/_pfaffian.py` | No automatic scalar reinterpretation in PfaffianPolicy; only supported nominal constructor inputs, if any, migrate. | Selector parse, float conversion, finiteness/tolerances and resource values. Missing scalar validation is a separately owned issue, not silently fixed by plain int annotations. |
| `phydrax/domain/_components.py` | Audit actual constructors/methods, not its raw guard count. | Boundary/Interior/geometry narrowing, AxisArray identity, selection polymorphism, normals/sdf/enforcement gates and coordinate semantics. |
| `phydrax/domain/_domain.py` | Audit selected methods only after resolving ModelBinding/DomainComponent/FunctionBinding import ownership. | Polymorphic Function/Model APIs, local cycle-breaking imports, callbacks, array-like inputs, model/port metadata. |
| `phydrax/integration/_api.py` | Audit IntegrationRealization precision/transformation constructor inputs. | Protected module functions including reduce/materialize APIs; DomainFunction/AxisArray/callable/constant dispatch, target-plan compatibility, transformation/precision evidence. |
| `phydrax/integration/_adaptive_callable.py` | Method/constructor sites only if present and eligible. | Existing adaptive_interval_callable/adaptive_triangle_callable bodies and callback-output/bounds/capacity contracts. |
| `phydrax/operators/_composition.py` | Audit methods/constructors from the appendix. | Source-addressed pullback body, substitution contents, target-domain/coordinate identity, callable evaluation. |
| `phydrax/operators/differential/_domain_ops.py` | Supported nominal method entries only. | StructuredDerivativeProvider and Binary/Unary/SwapAxes subtype route selection, derivative outputs, scientific operators and EvalKey/provider boundaries. |
| `phydrax/equations/_chemical_species.py` | ChemicalSpecies-style constructors: replace catalog nominal input guard where applicable. | Species/phase element checks and all shape/value/scientific identity contracts. |
| `phydrax/equations/_chemical_components.py` | Keep input/conversion distinction explicit; audit only independent nominal parameters. | Host-to-JAX conversion, Scope/parse, canonical catalog storage, composition/mass and species identity. |
| `phydrax/_admissibility.py` | AdmissibilityTransitionRequest evidence input guard is an eligible constructor candidate. | Region/epoch shapes, canonical identifiers, reason/status and admissibility semantics. |
| `phydrax/_array_tree.py` | Audit ArrayPyTreeSchema/PyTreeDef nominal constructor inputs. | Leaf element validation, dtype/size/path counts, ordering/uniqueness, schema identity, storage and untrusted admission. |
| `phydrax/_axis_factorization.py` | AxisFactorizedField plan input is a candidate. | Factor uniqueness, missing factors, semantic axes and contraction evidence. |
| `phydrax/_bvh.py` | Audit eligible constructor/method nominal BVH/policy entries. | Protected prepare/query free functions, dimension/rank/dtype, integer capacities, x64 and numerical overlap/query behavior. |
| `phydrax/_data_plane/_distributed.py` | DistributedIndexEpochPlan global_plan constructor guard is a candidate. | Intentional integer conversion/range, distribution/topology and revision semantics. |
| `phydrax/_execution_workset.py` | Audit PreparedExecutionWorksets/ExecutionRuntime/checkpoint and callable constructor/method entries. | Execution grouping, device/sharding, revision/fingerprint, PyTree and checkpoint ownership. |
| `phydrax/_exponential_family/_contracts.py` | Audit family/signature/statistics constructor/method guard sites. | Signature compatibility, evidence, numerical domain, and source-addressed functions. |
| `phydrax/_exponential_family/_conjugacy.py` | Same limited nominal migration. | Conjugacy/numerical scientific identities and constraints. |
| `phydrax/_exponential_family/_finite_support.py` | Audit family/plan nominal constructor/method sites. | Mean conversion, signature, finite support/domain behavior and protected functions. |
| `phydrax/lifecycle/_models.py` | Audit RevisionLineage/CheckpointManifest/ModelManifest/ResultManifest/ResultRevision constructor inputs. | Nested containers, canonical ordering, parent/payload/checksum/revision identity and uniqueness. |
| `phydrax/lifecycle/_archive.py` | Eligible class methods only. | Source-addressed migrate/rollback/create functions, decode-by-kind, exporter registry, payload/parents/checksum and domain corruption errors. |
| `phydrax/lifecycle/_restart_topology.py` | Eligible constructor/method plan/relation/policy sites only. | Protected restore functions, chunk contents, reader bytes/length/checksum, resource budgets, destination writes and rollback. |
| `phydrax/_model/_ports.py` | No broad guard removal. | PortProvider capabilities, returned ModelPorts type, port identity and evidence. |
| `phydrax/interchange/_inspection.py` | HostInspectionConversion nominal frame/report constructor entries can be audited. | Checker-only constructor overloads, __post_init__ normalization, arrays and target identity. |
| `phydrax/interchange/_report.py` | AdapterReport constructor/method nominal sites only. | Stage order/identity continuity/profiles/losses/capability negotiation and protected compose functions. |
| `phydrax/_external_resource.py` | Preserve untrusted admission; only an independent trusted method entry can qualify. | Bounded bytes/path readers, stable digest/resource/corruption errors. |
| `phydrax/interchange/_openpmd_mesh.py` | Individual supported method/constructor entries only. | HDF5 object kinds, standard fields, units/layout, budgets and optional providers. |
| `phydrax/interchange/_openpmd_pic.py` | Retain unresolved TYPE_CHECKING/provider-dependent boundaries unless safely resolvable in place. | Solver subtype dispatch, HDF5 validation, optional imports and protocol/format evidence. |
| `phydrax/_artifact_security.py` | Retain ArtifactManifest boundaries requiring TYPE_CHECKING/local imports unless a safe method-specific plan is proven. | Security/resource/manifest evidence; no eager cycle-inducing imports. |
| `phydrax/_external_runtime.py` | HostInferenceAdapter callable constructor presence may migrate if identical. | Input/output schemas, finite detached output, transport, host-only transformation refusal and binding identity. |

### 4.4 Facades and first-party consumers

`phydrax/__init__.py`, `phydrax/linalg/__init__.py`, `phydrax/domain/__init__.py`, `phydrax/integration/__init__.py`, `phydrax/operators/__init__.py`, and `phydrax/interchange/__init__.py` keep their existing export/lazy behavior. Add no re-export aliases.

For each migrated target, the actual reference set supplies the file-by-file caller list across `phydrax`, `tests`, `tools`, `benchmarks`, `examples`, `mkdocstrings_setup.py`, and `native/meshcore/python`. Record unchanged callers intentionally; do not invent caller changes to inflate the migration. A new runtime import must not turn a missing optional provider into an import failure.

### 4.5 PR #383 meshfree and adjacent owner matrix

All of these current owners are included in the appendix. `Candidate` below always means subject to both ordinary and compiler-source identity gates, runtime-resolution review, and accepted-domain/ordering proof.

| Current file | Eligible entry work to evaluate | Scientific/resource ownership retained |
|---|---|---|
| `phydrax/discretization/meshfree/__init__.py` | No guard migration; preserve lazy export map and TYPE_CHECKING aliases. | No eager import of all meshfree owners or optional providers. |
| `phydrax/discretization/meshfree/_types.py` | No nominal candidate from the conservative scan. | Canonical selector/type aliases; owner parse rather than duplicated selector sets. |
| `phydrax/discretization/meshfree/_neighbors.py` | Audit roles, not raw guard counts. | Bounded Morton nearest/radius preparation, active masks, duplicates, candidate gaps, pair/chunk/working-set refusal; no all-pairs fallback. |
| `phydrax/discretization/meshfree/_stencils.py` | Protect `prepare_local_stencils` source; audit actual constructor inputs separately. | Weighted-SVD and augmented PHS fits, polynomial functionals, row rank/condition/amplification/status and support capacity. |
| `phydrax/discretization/meshfree/_operators.py` | `MeshfreeOperator.__init__`: prepared-stencil nominal candidate. | Functional index, sparse source/target spaces, operator IDs, numerical application and adjoint. |
| `phydrax/discretization/meshfree/_stabilization.py` | `PreparedHyperviscosity.__init__`: plan nominal candidate. | Coefficient/order/power-iteration rules, dissipative form and bounded spectral evidence; no unconditional stability claim. |
| `phydrax/discretization/meshfree/_multilevel.py` | Protect `meshfree_multigrid_builder`; classify hierarchy constructors/methods. | Stable subset IDs, retained coarse nodes, polynomial transfer, affine reproduction, no-progress stop, nullspace semantics and transfer budgets. |
| `phydrax/discretization/meshfree/_surface_geometry.py` | Preserve source-bearing callback entries unless a genuinely independent nominal input is proved. | Constraint/metric/source programs, rank/orientation, projection/tube/support status; no ownership from coincident samples. |
| `phydrax/discretization/meshfree/_surface_quadrature.py` | Preserve current policy. | Supplied/tangent-Voronoi/normalized-density routes, positivity and explicit-area evidence. |
| `phydrax/discretization/meshfree/_surface.py` | Treat prepared_id/source_field/ambient_metric and source-bound callbacks as protected. | Exact field/metric program provenance and numeric revision, fixed support, anchored reference measures, trust/radius/partial coverage. |
| `phydrax/discretization/meshfree/_exterior.py` | No safe direct parameter candidate from the scan; classify actual owners. | Sparse moment/cochain/Hilbert binding, active/radius/capacity trust, sign admission and rollback-only refresh. |
| `phydrax/discretization/meshfree/_exterior_metric.py` | Preserve provider/metric admission. | Original moment audit, exact/relaxed slack, signed/nonnegative weights, sparse rank profile, amplification/status and derivative availability; no clipping or unused signed fallback solve. |
| `phydrax/discretization/meshfree/_exterior_transport.py` | `MeshfreeAdvection.__init__`: prepared-exterior nominal candidate. | Canonical incidence, outgoing CFL, extensive conservation ledger, metric admission and positivity refusal. |
| `phydrax/discretization/meshfree/_capacity.py` | Preserve capacity-map owner checks. | Compact/active spaces, increasing unique IDs, bucket exhaustion and refusal of zero-mass active rows. |
| `phydrax/discretization/meshfree/_shifting.py` | Protect `surface_relative_advection`. | Tangential/material-minus-mesh motion, geometry/support status and failed shift evidence. |
| `phydrax/discretization/meshfree/_resampling.py` | Preserve bounded resampling admission. | Quality/support/capacity decisions, deterministic epochs and refusal rather than automatic budget growth. |
| `phydrax/discretization/meshfree/_surface_transfer.py` | `SurfaceTransferPlan.from_stencils` and the closed relation union in __init__ require separate review. | High-order versus positive routes, coverage, weighted column conservation, constant reproduction, sign and duality evidence. |
| `phydrax/discretization/meshfree/_moving.py` | `MovingSurfacePlan.__init__`: epoch and audited nominal policy candidates; callback fields remain identity-sensitive. | Geometry callback output, complete histories, accepted-state commit/rollback, plan/epoch identity, all-history transaction refusal and conservation. |
| `phydrax/discretization/meshfree/_constitutive.py` | `EdgeFeatureField.__init__`, `EdgeModelLipschitzCertificate.from_model`; audit MonotoneEdgeConductance/LipschitzEdgeFlux union/model/certificate guards beyond the exact-spelling scan. | Feature scientific dimensions/parity/frame, canonical model family/certificate/activation, certified bounds, nonlinear law semantics and MODEL authority. |
| `phydrax/discretization/meshfree/_coverage.py` | Preserve assessment rather than treating it as type checking. | Covariance/quantile/support evidence and scientific refusal. |
| `phydrax/discretization/meshfree/_conservation_solve.py` | Protect `prepare_meshfree_conservation_solve` and bound nonlinear primal/adjoint source. | Conservation ledger, original residual, failed primal state and native adjoint status; failed-state/flux derivatives remain unusable. |
| `phydrax/discretization/meshfree/_profiles.py` | Preserve candidate/profile publication. | No release-status promotion or qualification inference from this typing migration. |
| `phydrax/discretization/_point_cloud.py` | Audit current constructor grammar only. | Canonical `stencil=LocalStencilPolicy(...)`, `neighbors=...`, support/masks/capacity and prepared-owner identity. |
| `phydrax/discretization/_point_cloud_pde.py` | PointDiffusionOperator/PreparedPointCloudPoisson constructor nominal candidates. | Collocated versus quadrature-adjoint dissipative form, positive coefficients, boundary lifting/Robin/Neumann gauge/compatibility, original-equation residual/status and numeric refresh. |
| `phydrax/discretization/_point_cloud_view.py` | Protect `prepare_point_cloud_field_reconstruction`. | Bounded BVH admission, polynomial versus explicit degree-zero Shepard, rank/conditioning and partial/complete query evidence. |
| `phydrax/linalg/_sparse_rank.py` | Protect `prepare_sparse_row_rank`. | Host-only tracer refusal, canonical sorted CSR and tolerance-defined profile; separate input/fill/elimination-work limits; no claim of algebraic rank or singular-value certification. |
| `phydrax/linalg/_preconditioning.py` | ProjectedPseudoinversePreconditionerBuilder/PreconditionerPlan/PreparedPreconditioner nominal constructor candidates. | Declared complete kernels, Euclidean coordinates, gauge/compatibility, factor rank/kernel residual, materialization budgets and numeric refresh. Preserve local/TYPE_CHECKING imports for subspace/rank/factorization types. |
| `phydrax/metrix/_ambient.py` | No exact-guard candidate from the scan; preserve callback/source ownership. | Regular-level-set normal projection, SPD/rank/status and local geometric admissibility. |
| `phydrax/solver/coupling/_meshfree_components.py` | MeshfreeComponent constructor's native owner/operator/reconstruction union can be reviewed. | Exact owner/query/source revision, native diagonal measures, spaces/coordinate transpose, constrained pullback and homogeneous kernels. |
| `phydrax/solver/coupling/_surface_exchange.py` | LangmuirAdsorptionFlux/SurfaceExchangeLaw/MeshfreeBulkSurfaceMethod constructor nominal candidates; protect `_admit_query`. | Complete scalar query, constant reproduction, exact coordinate transpose, explicit positive Shepard deposition, point enumeration/measures, epoch/lag/displacement and nonnegative metric admission. |
| `phydrax/solver/_fixed_step.py` | FixedStepProblem/LearnedStepCorrection constructor and FixedStepRolloutPlan.rollout candidates only after callback/source gate; protect solve_fixed_step. | FixedStepResult.evidence, transform admissibility, status/retry/replay, accepted-state rollback and frozen-decision sensitivity. |
| `phydrax/nonlinear/_domain_cond.py` | No wrapper/primitive/rule changes. | Lane-local refusal, known/unmapped primal outputs, residual reuse, batching/JVP/transpose/partial-evaluation/effect order. |
| `phydrax/nn/models/_input_convex.py` | No unrelated type-checking rewrite. | Negative-offset positive-transform refusal and certification of input convexity. |
| `phydrax/optim/_programming/_native_conic.py`, `_native_hsd.py` | No numerical owner changes for guard cleanup. | Matrix-free sparse Newton actions and original-coordinate KKT/complementarity termination. |
| `phydrax/qualification/_builtin_catalog.py` | Preserve the post-PR catalog. | Meshfree unreleased candidate profiles and existing scientific/resource release gates. |

The four exact protected meshfree package free functions are prepare_local_stencils, meshfree_multigrid_builder, surface_relative_advection, and prepare_meshfree_conservation_solve. Compound nominal/union guards not counted by the conservative scanner still require a disposition; the seven counted method sites are not the full eligible set.

Never restore or alias the deleted `_local_polynomial.py`/`meshfree.py` implementation or the removed MeshfreeStencilPlan, prepare_meshfree_operator, MeshfreeMethod, MeshfreeOperatorKind, PreparedMeshfreeOperator, MeshfreeReproductionEvidence, PointStencilReport, DissipativePointDiffusion, solve_point_cloud_poisson, PointConormalInterface, or DistributedPointPartition APIs. Do not reintroduce PointCloudPlan(degree=..., neighbor_count=..., condition_limit=...) or dense Poisson/dissipative fallbacks.

Preserve the lazy meshfree and coupling facades, their canonical __all__ order and post-PR public catalog. `_surface.py` and `_surface_exchange.py` intentionally resolve some exterior types via TYPE_CHECKING plus local imports; do not eagerly load them for a checker. Benchmark/example providers remain in their host owners, not package startup.


## 5. Tooling and coverage ratchet

### `tools/audit_contract_candidates.py`

Extend the existing read-only audit rather than adding another competing validator/audit convention:

- Keep current structural candidate ranking intact.
- Add a separately selected signature/migration report, with AST candidate discovery and actual selected-boundary coverage.
- Use authoritative public exports and explicit method targets; do not treat a spelling match as scientific identity or publicness.
- Report canonical file/symbol/parameter, body ownership, enclosing protected source function, annotation category/resolution, decorator placement, and migrated/retained reason.
- Detect duplicate checked attachment, selected boundaries with no effective checks, and leftover exact guard duplication at checked methods.
- Do not flag scalar/capability/decoder checks as violations merely because they use isinstance.
- Preserve deterministic ordering and the existing CLI behavior. New options must be documented; command names below that add signature reporting are proposed, not current invocations.

A zero-unclassified-candidate ratchet is required. A zero-isinstance ratchet is explicitly forbidden by this plan.

### `tests/unit/test_contract_declarations.py`

Keep the existing stored-field opt-in and runtime-resolution audit. Add selected-signature coverage/resolution checks as separately collectable invariants. Exclude `checked` from `_VOCABULARY_NAMES`' helper exports so a callable utility is not mistaken for a type form.

This declaration audit must not instantiate every class or call every product function merely to resolve a plan. Preserve optional dependency ownership and deliberately static-only conversion/provider annotations.

### Existing API/static tools

- `tools/generate_public_api_manifest.py`: run existing generator; no guard inventory is placed in the public manifest.
- `tools/check_public_api_manifest.py`: existing drift gate.
- `tools/check_import_boundaries.py`: existing docs/examples boundary gate.
- `tools/audit_selectors.py`: keep zero findings; selector canonicalization is not removed.
- `tools/check_typing.py`: retain pinned ty and Ruff ANN checks, all first-party coverage, no new checker configuration or broad Any/suppressions.
- `tools/check_installed_typing.py`: retain its wheel-install consumer workflow; add public decorator consumer coverage through fixtures, not a new runner.

## 6. Tests: meaningful contracts, file by file

Do not delete tests merely because they contain a ty ignore or a TypeError assertion. Do not create one decorator-forwarding test per migrated class. Keep consumer-visible failure, normalization, identity, status and lifecycle coverage.

### 6.1 New `tests/unit/typing/test_signature.py`

Own the new boundary scenarios with deterministic local fixtures:

1. Nominal object kind, valid subclass, optional None, closed nominal union, and wrong kinds.
2. Callable presence with callable objects and wrong non-callables; output/signature semantics demonstrably remain owner-owned.
3. Two canonical array/metadata parameters bind one nominal dimension; mismatch fails before body work; successive calls do not retain bindings; failed union alternatives roll back.
4. Native array/key kinds versus dtype/rank/extent errors have the correct TypeError/ValueError categories. Reuse a small representative matrix, not the full existing form test matrix.
5. No implicit host/list-to-JAX conversion: canonical array inputs refuse host values while explicit as_array remains the accepted path.
6. Plain scalar/conversion annotations preserve owner Integral/Real/NumPy admission and normalization. Include bool exclusion where the owning validator already enforces it, not because the decorator redefines int.
7. Literal/Enum annotations do not preempt owner parse; np.str_ accepted by the owner becomes its canonical stored string, and invalid selectors retain the existing category.
8. Python binding failures occur before nominal/structural checks; cover positional-only, keyword-only, default, duplicate and missing arguments with diagnostic parameterization.
9. Selected forward annotations resolve lazily in the correct defining module/class; unresolved required names fail with qualified ownership; same-name distinct classes do not share a stale plan.
10. Concurrent first compilation does not leak Scope/state or mix plans across calls.
11. Descriptor behavior for an instance method, classmethod, staticmethod and selected property getter where meaningful; receiver exclusion does not turn a free parameter named self into a bypass.
12. A custom-JVP/VJP primal keeps its registration interface and produces the independently known derivative; derivative-rule tangents are not treated as primal inputs.
13. Body exceptions and result objects propagate unchanged after a valid entry; no fallback or return checking is introduced.
14. Unsupported native placement fails rather than silently bypassing shape semantics; an ordinary static-only mixed union retains its owning validation.
15. Invalid nominal input refuses before a real side effect/dereference while valid input exercises the actual boundary result. Do not test only that a wrapper calls a mock.

Use normal parameterization and existing `_support` invariants when reusable. No soft assertions, broad loops of unrelated scenarios, source-text assertions, annotation-spelling tests, or incidental exact wording pinning.

### 6.2 Existing runtime files

| File | Change or intentional preservation |
|---|---|
| `tests/unit/typing/test_forms.py` | Keep full stored-field/parse grammar tests; extend only for input-vs-field distinctions not covered by the new boundary scenarios. |
| `tests/unit/typing/test_scope.py` | Keep Scope and rollback property coverage; avoid repeating it solely through another wrapper. |
| `tests/unit/typing/test_convert.py` | Preserve one-time conversion, dtype category, tracer and transfer-guard behavior; add only a missing consumer-visible input conversion regression. |
| `tests/unit/typing/test_strict_integration.py` | Add audited checked custom/inherited constructor scenarios; converters run once; __check_init__ sees converted fields; existing post-construction field order and independent nested scopes remain. Preserve tree_at/tree_map explicit validation, recipe tampering and JIT/vmap paths. |
| `tests/unit/test_strict.py` | Existing abstract/final/immutability/static-signature behavior remains. Add a checked custom constructor refusal scenario only if not owned by integration tests. |
| `tests/unit/test_contract_declarations.py` | Separate new checked-target declaration/resolution audit from existing field opt-in audit. |
| `tests/unit/test_identity.py` | Preserve ordinary callable rules and PR #383 program/capture/imported-global/custom-JVP/partition/primitive/effect/placement contracts. Compare field IDs and execution metadata for genuinely eligible migrated boundaries. Confirm checked free callbacks remain opaque and unowned compiler wrapper state is not silently admitted. Never re-pin a source hash to hide drift. |
| `tests/unit/test_array_tree.py` | Preserve tree inventory, paths, dtype/storage and canonical IDs after eligible schema constructor migration. |
| `tests/unit/test_array_archive_security.py` | Preserve resource/admission/security errors; no reinterpretation as signature validation. |
| `tests/unit/linalg/test_operator_contracts.py` | Exercise migrated operator constructors with valid/invalid kinds plus independent space/capability/adjoint/materialization contracts. |
| `tests/unit/linalg/test_checked_solve_contracts.py` | Preserve solve status, finite-input behavior, capability, residual and differentiation evidence. |
| `tests/unit/ml/test_artifacts.py` | Preserve exact registration, deterministic recipe, tampering/preallocation/leaf inventory and complete restore validation. |
| `tests/integration/test_public_api_manifest.py` | New public checked export is discovered by the existing manifest contract; preserve lazy-export determinism. |
| `tests/integration/test_import_boundaries.py` | Public docs/import paths remain valid; optional provider loading is unchanged. |

For each other migrated owner file, use its actual corresponding consumer test/reference set. The audit disposition must identify it; do not guess a test filename from the owner path. Add a new permanent case only for a plausible boundary/ordering/lifecycle bug not already covered.

### 6.3 Static and installed fixtures

- New `tests/typing/cases/checked_signatures.py`: preserve ParamSpec, keyword-only/positional-only/defaults, methods, generics and return types with assert_type; deliberate invalid calls use exact line-local ty rules and reachable misuse is paired with runtime refusal coverage.
- `tests/typing/cases/typing_forms.py`: remain the vocabulary base-type fixture; update only if the new public API examples require it.
- `tests/typing/cases/strict_constructors.py` and `inherited_constructors.py`: check decorated custom constructors retain their true generated/custom/inherited signatures. Do not expose a metaclass (*args, **kwargs) signature or add ignores to conceal it.
- `tests/typing/cases/type_forms.py`, `boundary_forms.py`, `exterior_forms.py`, `exterior_pde.py`, `operator_valued_forms.py`, and `vem_forms.py`: run through the existing fixture runner; modify only genuinely affected calls.
- `tests/typing/installed/consumer.py`: import checked from the installed wheel and assert preserved public callable/constructor types outside the source checkout.
- `tests/typing/test_typecheck.py`: current sorted fixture discovery remains the runner; no second static checker or new runner.

### 6.4 Coverage migration and deliberate faults

Record old guard -> new structural owner -> consumer scenario mapping. If tests are materially consolidated, capture affected-module coverage before/after and keep the ratchet. Do not present lower test/line counts as preservation evidence.

Before deleting duplicated permanent tests, use temporary deliberate faults to demonstrate adequacy at shared engine branches: omit nominal rejection, leak a Scope across calls, use stored-field annotations on raw converting input, and wrap outside custom_jvp. The relevant consumer scenarios must distinguish these faults. Remove fault edits afterward; do not retain source-text or wiring tests.

### 6.5 PR #383 existing consumer proofs

Use these existing tests when the listed owner is actually migrated. The final global suite, when required, already collects them; do not separately rerun every group merely to duplicate evidence.

| Affected owner | Existing consumer files and contracts |
|---|---|
| Point/PDE/stencil/hyperviscosity | `tests/unit/discretization/meshfree/test_point_cloud_poisson.py`, `test_hyperviscosity.py`, `test_surface_operators.py`, `test_surface_geometry.py`, `test_surface_quadrature.py`; `tests/unit/meshing/test_assembly.py` for the current neighbors caller contract |
| Sparse rank/metric/transport | `tests/unit/linalg/test_sparse_rank.py`; `tests/unit/discretization/meshfree/test_exterior_metric.py`, `test_exterior_transport.py`; preserve original-moment/status, sign and independent budget refusal |
| Moving geometry/epochs/history transfer | `tests/unit/discretization/meshfree/test_moving_surface.py`, `test_surface_transfer.py`, `test_surface_resampling.py`, `test_surface_shifting.py`; every saved history must transfer or the entire transaction returns the original state |
| Constitutive certification and failed-state adjoints | `tests/unit/discretization/meshfree/test_constitutive_laws.py`, `test_conservation_solve.py`; `tests/unit/nn/test_input_convex.py`; `tests/integration/test_meshfree_learned_flux_training.py`; preserve parity/frame certification and failed forward/adjoint/training refusal |
| Coupling/source/revision admission | `tests/unit/solver/coupling/test_meshfree_components.py`, `test_surface_exchange.py`; `tests/integration/test_meshfree_bulk_surface_workflow.py`; preserve stale numeric revision/source rebinding, overcapacity rollback, signed-query positivity refusal and ambient-dimension refusal |
| Fixed-step callback and transform semantics | `tests/unit/solver/test_fixed_step.py`, `test_learned_step_correction.py`; preserve evidence publication, inherited/custom signatures, callback ordering, rollback and native sensitivity |
| Compiler/native callback identity | `tests/unit/test_identity.py`; preserve static versus dynamic capture, custom-JVP selectors/recursive source, imported helper bindings, partitioned closures, stack/batch axes, tree static semantics, primitive/effect/placement refusal and declared derivative refusal without evaluating it |
| Shared guarded nonlinear semantics | The actual sparse derivative/failed-state reference set of `_domain_cond.py`; include relevant meshfree conservation/training scenarios rather than adding a wrapper that probes both branches |

The direct nonlinear/callback proof set includes `tests/unit/solver/coupling/test_execution_contracts.py::test_members_with_different_custom_derivative_rules_never_share_lanes` and `::test_members_with_different_host_callbacks_never_share_lanes`; `tests/unit/solver/test_trial_validity.py::test_explicit_jacobian_is_not_called_on_rejected_initial_state` and `::test_mapped_newton_never_evaluates_invalid_residual_or_jacobian_lane`; and `tests/unit/solver/test_dae_trial_domain.py` for initialization/event/replay, JIT/vmap, JVP/grad and Python-scalar-time tracing.

Use `tests/integration/test_meshfree_bulk_surface_workflow.py::test_surface_source_program_rebinding_refuses_stale_reconstruction` as the direct source-identity consumer: equal sample points, labels and operators are insufficient after constraint-program rebinding. Existing moving history refusal and fixed-support refresh/JVP cases remain the epoch/geometry proof.

For the new feature, add only missing meaningful identity-admission scenarios to `tests/unit/test_identity.py`: a checked stateless callback is not made transparent through __wrapped__; compiler traversal does not silently admit checker-private closure/cache state; a custom-JVP/imported-helper binding is not collapsed to its pre-wrapper declaration; and a native closure-converted program retains its declared static/dynamic capture distinction. Choose fixtures with a definite native admission/refusal contract—do not accept 'either a different ID or an exception' as a permanent assertion, and do not re-pin old IDs. Existing foreign/bound callback, effect and derivative-refusal scenarios are reused rather than copied.


Beyond ordinary object IDs, compare canonical execution_metadata_payload records for any eligible migrated boundary entering surface source provenance, a native custom-JVP or callback, a closure-converted program, or static PyTree metadata. If admission/payload changes, keep the original guarded source; do not re-pin compiler IDs or add a generic checker adapter.

A new permanent regression is warranted only where the migration creates an uncertain interaction not covered by these existing scenarios: checked input refusal before a real callback is invoked, unchanged converter/owner order, preserved partial coverage, or a retained failed-state sensitivity boundary. Do not duplicate the scientific Q1–Q8 matrix in typing tests.


## 7. Smoke proof and benchmarks

### 7.1 Baseline before implementation

In the fresh worktree, before changing runtime code:

- Exercise a nominally guarded constructor, an inherited converting constructor, an array/metadata boundary, a transformed module with explicit validate, and a callable identity consumer.
- Record admitted/refused inputs, single-fault exception categories and meaningful ordering, canonical fields, PyTree structure, and consumer IDs.
- Record existing typing benchmark results in a temporary baseline artifact using the same environment and capacities as the post-change run. Do not overwrite the tracked canonical JSON with the baseline.

These are future actions, not performed checks.

### 7.2 Actual post-change smoke

Use a throwaway public-API driver that imports the installed/source public surfaces, constructs a real linalg operator/problem, exercises a selected checked method, solves/evaluates it, and observes numerical output plus status/evidence. Do not use mocks as the runtime path.

The driver also demonstrates:

- wrong nominal kind refuses before body work;
- owner conversion accepts the same host/NumPy input and produces the same canonical fields;
- owner selector parsing retains accepted NumPy spelling;
- shared dimensions reject an incompatible second argument;
- JIT and one derivative path return independently expected values;
- tree_at/tree_map require explicit validate and invalid transformed state is rejected;
- recipe restoration still validates the full result;
- field-based object/plan/callable IDs and exact compiler program/source/capture payloads match the post-PR baseline for identical valid inputs;
- protected source-addressed callable payloads match the baseline; ordinary opaque callbacks still refuse missing explicit IDs, and compiler-bound callbacks retain native-owner admission rather than accepting arbitrary checker closure metadata.

Remove the driver after observed proof. Report only scenarios actually exercised at implementation delivery.

### 7.3 `benchmarks/typing_structural_validation.py`

Extend the existing benchmark rather than adding a second timing framework. Reuse `benchmarks/_runtime.py` and `benchmarks/_io.py` unchanged unless an actual missing shared invariant is demonstrated.

Separate phases:

1. import/package startup in isolated subprocess;
2. cold signature/annotation plan compilation;
3. warmed nominal/callable-only checked calls;
4. warmed shared-dimension metadata calls;
5. checked custom constructor versus equivalent original guarded constructor;
6. existing generated opted/unopted construction and explicit validate;
7. trace, lowering, compilation, first synchronized execution, warmed compiled execution;
8. host plan cache and logical retained metadata bytes;
9. compiler temporary/output/code bytes and logical output bytes where available.

Vary controlling capacities: parameter/field counts and number of distinct plans, using a bounded small ladder plus the existing full ladder. Vary representative array extents separately: metadata checking must not scale with array element count or transfer array values. Include repeated prepared use to distinguish cold plan setup from warmed checks.

Compare against an actual manual guard baseline, not only an unvalidated identity function. Benchmark invalid paths separately if they expose a material diagnostic cost, but do not optimize away failure evidence.

Do not add return-checking, canonicalization or generalized container benchmarks for features this plan excludes. JAX equation/compiled-byte comparisons belong in this evidence, not source/wiring unit tests. A host-only checker should leave warmed compiled numerical work unchanged; prove that rather than claiming it from design.

Regenerate `benchmarks/typing_structural_validation.json` only after post-change measurements. Keep current canonical record conventions; no schema-version/generation fields.

### 7.4 Downstream benchmarks

When affected linalg constructors/methods are in a measured path, run existing appropriate rows from `tools/linalg_advanced_benchmarks.py`, `tools/exterior_linalg_benchmarks.py`, or `benchmarks/linalg_inverse.py`. These already provide prepared/warm or phase-separated evidence. Do not rewrite unrelated `tools/linalg_benchmarks.py` timing contracts solely because its cold phase is combined.

For actually migrated PR #383 owners, reuse the corresponding phase-separated driver:

| Changed owner | Driver and relevant phases |
|---|---|
| Point/neighborhood/stencil/PDE | `benchmarks/meshfree_scaling.py`: neighbor/stencil preparation, lower, compile, first/warm application and adjoint |
| Exterior/metric/rank/transport | `benchmarks/meshfree_exterior.py`: neighborhood, moment/metric/cochain preparation, numerical refresh, action and adjoint |
| Hierarchy/projected coarse solver | `benchmarks/meshfree_multilevel.py`: hierarchy/provider setup, original fine-system residual, warmed action, adjoint and numeric refresh |
| Surface/moving/epoch transfer | `benchmarks/meshfree_moving_surface.py`: surface stencil/refresh, moving assembly, geometry refresh, epoch-transfer preparation/commit, shift/status, action and adjoint |
| Coupling or constitutive source/adjoint interaction | Existing integration workflow smoke; select only Q7 or Q8 in `tools/meshfree_qualification.py` if that scientific boundary was changed |

Select Q2/Q3/Q4/Q5/Q6 only when the respective elliptic/multilevel/exterior/surface/epoch owner was changed. Do not run all Q1–Q8 campaigns for a nominal decorator change. Use actual compatible before/after records; do not invent a baseline or convert logical/compiler memory estimates into a physical peak-memory claim. Preserve unexecuted release gates and candidate status.

The existing `benchmarks/meshfree_moving_surface.json` reports logical retained bytes as unavailable where callbacks hold operator arrays beyond the runtime object walker. Keep that limitation explicit. New host checker metadata/cache measurements must be reported separately and cannot be used to claim all callback capture memory or physical peak resident memory was counted.



No source-faithful numerical rewrite is planned. If a real benchmark regression requires changing runtime preparation, compiler lowering, vectorization, or linalg semantics, identify that as a separate owner extension and revise the plan before expanding scope.

## 8. Documentation and generated files

| File | Required update |
|---|---|
| `docs/guides_typing.md` | checked API, supported/static-only input grammar, no-conversion/no-return policy, per-call Scope, descriptor/JIT placement, custom-constructor versus stored-field distinction, runtime resolution, and existing callable identity/opaque callback implications |
| `docs/api/typing.md` | Public checked mkdocstrings entry alongside parse/conversion/validate; explain partial enforcement precisely |
| `docs/guides_testing.md` | Boundary/installed fixture ownership, real smoke requirement, minimal affected selection and cross-cutting full-suite rule, benchmark phase evidence |
| `docs/api/phydrax.md` | Only a discoverability cross-link if needed; no root alias |
| `docs/api/linalg.md` | Only affected example/benchmark explanation; no unrelated benchmark vocabulary changes |
| `docs/data/public_api.json` | Regenerate via existing generator after typing.__all__ changes; do not hand-edit |
| `CHANGELOG.md` | One Unreleased entry: native checked constructor/method boundaries, precise preserved conversion/scientific/identity behavior, constrained free-function cleanup, and enumerated intentional entry-error precedence changes |
| `docs/plans/runtime-contracts.md` | This execution contract; revise only for explicit design findings/user decisions, not to erase failed evidence |
| `docs/plans/runtime-contract-files.tsv` | Baseline per-file checklist; refresh against implementation tree before migration, with canonical ordering and coherent source hashes |

`mkdocs.yml` already has Typing guide/API entries; no new navigation item is needed. `mkdocstrings_setup.py` already renders type aliases; change only for an observed decorator-signature rendering defect. No typing-specific capability/qualification manifest was found; do not alter capability_closure/application qualification records for an API-only addition.

Examples that teach checked boundaries should define new illustrative constructors/methods and retain explicit ArrayLike conversion and selector parse. Do not rewrite existing source-addressed scientific example functions just to make the style uniform.

Preserve post-PR meshfree docs (`docs/guides_meshfree.md`, `docs/api/discretization/meshfree.md`, `docs/api/solver/coupling.md`) and the new examples. Update them only if an actual admitted input/import changes; do not revive removed APIs. The public API generator must retain current meshfree exports while adding checked, and capability/application portfolio records remain the current candidate catalog rather than new typing qualification claims.


## 9. Execution phases and integration ownership

### P0 — Worktree, baseline, and classified map

A future //code request creates a fresh worktree from the then-current branch under `/Users/lgleyzer/PHYDRA/phydra-labs/.worktrees/`, unless //code-- explicitly selects the current worktree. Copy the authoritative ignored AGENTS.md into the new worktree before any implementation work. Use a neutral name such as `runtime-contracts`; do not name the branch/worktree after another library.

Refresh the static checklist, actual public/member/reference set, constructor ownership, and identity-protected functions. Capture the baseline smoke/benchmark/coverage evidence above. Select exact targets and retained reasons. Do not implement on the parent branch.

### P1 — Shared owner and one boundary API

Implement `_typing_plan.py` additions, `_typing_signature.py`, and typing.checked. Complete the supported/static-only grammar, binding, annotation resolution, caching, descriptor and transform semantics. Existing stored-state compiler and Strict lifecycle remain canonical.

One integration owner controls these shared files. No parallel agent independently edits the same contract variants or wrapper APIs.

### P2 — Complete eligible caller migration

Partition the entire classified method/constructor set by dependency/subsystem owner, not by arbitrary file count. Independent slices can run in parallel after the core API is fixed:

- linear/numerical owners and execution;
- domain/operators/equations/integration;
- lifecycle/model/interchange/resource owners;
- remaining application/discretization/solver/material/scientific domains from the appendix.

Each slice gets the exact wrapper API, grammar, identity decision, retained categories and test ownership. Slices may edit only their assigned files and must not run tests/builds/formatters mid-flight. Shared public facade/static test/doc files have one integration owner. The appendix's remaining groups are mandatory, not deferred 'follow-up'.

### P3 — Contract tests, audit, docs and generated API

Update meaningful tests/static fixtures and the existing candidate/declaration audit. Complete every disposition and every affected first-party caller. Update documentation and regenerate public API data after smoke proof. Remove superseded guards/imports and genuinely obsolete incidental-wording tests; retain semantic failures and accepted-domain tests.

### P4 — Integrated verification and evidence

Run the actual post-change surface smoke, selected phase-separated benchmarks, configured lint/format/type/documentation/API gates, and the minimal conservative affected test set. The user explicitly directed that the full suite be killed and not restarted.

The implemented checker is opt-in: Strict construction, stored-field semantics, test configuration, shared fixtures, and numerical kernels remain unchanged. Verification therefore covers the checker/Strict/declaration/identity group, direct migrated consumer and refusal scenarios, compiler-callback and source-rebinding contracts, public API/import boundaries, static fixtures, and the first-party typing gate. Do not replace this selection with all numerical-reference campaigns or provider qualifications merely because many files changed. Record actual nodes and results; the interrupted full-suite run is not passing evidence.

## 10. Planned verification commands

These command shapes are based on existing tool interfaces and were not run during planning. Use the actual worktree environment, optional-provider boundaries and configured formatter. Avoid duplicate test runs of the same contracts merely to produce more output.

```console
# Full first-party typing and annotation gate
uv run --extra qa python tools/check_typing.py check

# Installed-wheel public typing consumer
uv run --extra qa python -m tools.check_installed_typing

# Generated API and documentation boundary gates
uv run python tools/generate_public_api_manifest.py
uv run python tools/check_public_api_manifest.py
uv run python tools/check_import_boundaries.py
uv run python tools/audit_selectors.py
uv run --extra docs mkdocs build --strict

# Necessary checker, construction, identity, public API and static fixture checks.
uv run --extra tests --extra qa pytest -n auto \
  tests/unit/typing tests/unit/test_strict.py \
  tests/unit/test_contract_declarations.py tests/unit/test_identity.py \
  tests/unit/linalg/test_operator_contracts.py \
  tests/unit/linalg/test_checked_solve_contracts.py \
  tests/integration/test_public_api_manifest.py \
  tests/integration/test_import_boundaries.py tests/typing/test_typecheck.py

# Also run the selected migrated consumer/refusal and callback/source-identity
# scenarios documented in section 13, not the full suite.

# Existing typing benchmark smoke after implementation
uv run python benchmarks/typing_structural_validation.py \
  --quick --repeats 20 --output benchmarks/typing_structural_validation.json

# Full capacity ladder for the actual performance comparison
uv run python benchmarks/typing_structural_validation.py \
  --repeats 50 --output benchmarks/typing_structural_validation.json

# Representative existing downstream benchmark, when its owners were migrated
uv run python tools/linalg_advanced_benchmarks.py --smoke
uv run python benchmarks/linalg_inverse.py \
  --sizes 2 8 --batch-sizes 1 2 --warmup 0 --repeats 3
```

The candidate audit's signature/disposition command is a new interface to implement in P3; do not present it as an existing executable command. Existing structural ranking remains `python tools/audit_contract_candidates.py` with its current flags.

The installed-typing tool determines its actual interpreter workflow; reconcile available interpreters with the current package Requires-Python contract. Do not claim Python versions were exercised merely because the policy mentions them.

## 11. Acceptance criteria

The deliverable is complete only when all of the following hold:

1. There is one documented `phydrax.typing.checked` API and no import-hook/dependency/alternate-spelling convention.
2. Supported input contracts are declared once and checked before body assignment/dereference at every migrated boundary.
3. Existing stored-field grammar, constructor/converter/__check_init__/freeze order, field order and PyTree structure are unchanged.
4. No new parameter coercion, selector rewrite, hidden transfer/synchronization, numerical operation, materialization, or return checking occurs.
5. Scalar/Integral/Real/NumPy acceptance and owner selector canonicalization remain as before.
6. Supported cross-argument dimensions share one fresh Scope; calls, nested modules and restored objects do not accidentally share it.
7. Binding errors, deterministic input order and type/value categories are proved; intentional entry-precedence changes are enumerated.
8. Runtime forward resolution, inheritance, descriptors, JAX transforms, static signatures and installed-wheel consumers work on exercised surfaces.
9. Existing scientific object/plan/callable and compiler-program/source identities, deterministic RNG, provenance and persistence payloads remain unchanged for identical admitted inputs. Protected function/method/rule bodies, captures, transformations and imported helper bindings are untouched; opaque/unowned-state refusal is not weakened.
10. Callback outputs, native status/rank/conditioning/convergence/resource evidence, guarded nonlinear differentiation, complete epoch rollback, FixedStepResult evidence, capability gates, dispatch, decoder/security admission and transformed/restore validation still reach consumers.
11. Every current inventory candidate has a migrated or explicit retained disposition; no remaining actionable method/constructor guard is silently deferred.
12. No duplicate nominal guard remains at a migrated boundary unless it protects a distinct conditional/dynamic invariant identified in the map.
13. All affected first-party callers, static fixtures, public examples/docs and generated API records are updated or explicitly unchanged.
14. Permanent tests cover plausible consumer-visible faults, not wrapper wiring/source text/incidental messages; any material consolidation has a coverage map/ratchet and fault adequacy evidence.
15. Actual runtime smoke, relevant benchmark phases/capacity scaling and the applicable final test/type/lint/docs/API gates have observed output. Benchmarks do not conceal warmed host overhead with JIT-only timing.
16. No historical identity fallback, compatibility wrapper, phantom source payload, schema-generation field, placeholder or deferred implementation remains.

## 12. Principal risks and chosen controls

| Risk | Control |
|---|---|
| Literal canonicalization is rejected early | Literal/selector inputs stay owner-parsed; signature checker does not rewrite inputs |
| Scalar types become unintentionally strict | Plain scalars remain static-only; existing validators retain acceptance and conversion |
| Stored field contracts reject raw constructor inputs | Check only audited input annotations; generated constructors and converters remain in existing lifecycle |
| TYPE_CHECKING resolution creates cycles/provider loading | Lazy selected resolution, explicit retained boundaries, no blanket runtime imports |
| A wrapper or removed guard changes callable IDs | Protect source-addressed functions; no identity-engine exemption or code-payload preservation trick |
| A checker leaves equations unchanged but alters native source identity | Compare actual compiler/source/capture/transform payloads and admission; retain compiler-bound method/rule guards and imported helper bindings |
| Generic callback checking hides failed meshfree state or epoch refusal | Preserve callback-output evidence, failed-state JVP/VJP/training refusal, all-history transactional transfer and FixedStepResult.evidence |
| Nominal checking hides invalid transformed contents | Preserve validate/validate_tree at consumer and restore boundaries |
| Wrapper breaks descriptors/custom differentiation | Standard inner decorator placement, truthful signatures, independent derivative scenarios |
| Checker cost exceeds hand-written guards | Cold/warm host measurement, parameter-capacity ladder, cached slot plan and nominal no-Scope path |
| Cache retains dynamic types/arguments | Object-identity keys, weak-lifetime review, no bound argument/Scope/numerical state caching |
| Guard counts are mistaken for coverage | Per-symbol old-to-new disposition and real consumer verification |
| Concurrent work invalidates the checklist | Source hashes, refresh/read before edits, no reverting unrelated source |
| A large rollout becomes a partial scaffold | Core contract fixed first, all eligible appendix groups assigned, zero unclassified/disposition gaps at delivery |

The resulting code is cleaner because the same supported kind contract is no longer maintained independently in both an annotation and a guard. The checks that remain describe real scientific, lifecycle, conversion, identity, provider, or security ownership—and remain visible for that reason.

## 13. Implementation outcome

- `phydrax/_typing_plan.py` adds `NominalContract`, `CallableContract`, and `compile_input_form` beside the unchanged stored-field `compile_form`; the one `check` dispatcher owns every violation.
- `phydrax/_typing_signature.py` owns lazy per-function signature plans, a nominal fast path, one fresh `Scope` per call when a dimension binds, Python-binding precedence on refusal, and the explicit `phydrax.typing.checked` decorator. `_strict.py` is unchanged.
- Migration was runtime-verified per guard: the enclosing method's complete plan had to compile, the guarded parameter had to compile to exactly the guard's nominal class, the guard had to be an unconditional top-level statement with no earlier rebinding or exit, and checking the rest of the signature could add only nominal or callable contracts.
- [runtime-contract-dispositions.tsv](runtime-contract-dispositions.tsv) maps all 11,058 non-primitive exact-annotation guards of the baseline: 6,086 migrated to `checked` constructors and methods; 4,515 retained in source-addressed module-level functions; 217 retained because the annotation is static-only under `checked` (selectors, protocols, conversion types); 109 retained because the signature does not resolve at runtime; 96 retained because checking the signature would add array or metadata enforcement to other arguments; 24 retained in compiler-source-protected owners; 6 retained at identity or decorator boundaries; 5 retained because the guard is conditional or follows an early exit.
- `tools/audit_contract_candidates.py --signatures` reports zero redundant guards at checked boundaries, and the repository declaration test compiles every checked boundary and refuses one without an effective check.
- Verification follows the user's narrowed test policy: the full-suite process group was killed. The focused core group, targeted consumer/refusal nodes, cache-lifetime regression, public API/import gates and static typing contracts replace a whole-suite run; only observed results are reported at delivery.
- The signature registry uses weak keys and weak plan values so a cached nominal owner or original function's globals cannot keep dynamic classes and checked wrappers alive. A dynamic-module lifetime regression exercises compilation, removal and collection.
- Meshfree examples from PR #383 now import the same exported objects through their canonical public facades. Their computation bodies and scientific callback bindings are unchanged.

### Selected migrated consumer and refusal nodes

The additional consumer selection is bounded to the following nodes; parametrization may collect more than one item per node. It supplements the typing/Strict/declaration/identity group and the API/static fixture gates, without running unrelated numerical qualification campaigns:

```text
tests/unit/linalg/test_operator_contracts.py::test_operator_contracts_scenario_1
tests/unit/linalg/test_operator_contracts.py::test_block_actions_preserve_columns_and_fusion_declarations
tests/unit/linalg/test_checked_solve_contracts.py::test_checked_contracts
tests/integration/test_coupled_rom.py::test_reduced_component_requires_a_component_galerkin_model
tests/unit/applications/nucleic_acid_biophysics/test_strand_displacement_likelihood.py::test_reporter_calibration_isolated_and_full_trace_prediction_is_condition_bounded
tests/unit/applications/test_black_hole_identity_cutovers.py::test_black_hole_identity_cutovers_scenario_1
tests/unit/applications/test_cardiovascular_modalities.py::test_cardiovascular_modalities_scenario_1
tests/unit/applications/test_plane_stress.py::test_coupled_plane_contracts
tests/unit/applications/test_semiconductor_detector.py::test_semiconductor_detector_scenario_1
tests/unit/control/test_belief_lqg.py::test_belief_lqg_scenario_2
tests/unit/discretization/test_finite_volume_stage_metrics.py::test_stage_contracts
tests/unit/optim/test_mirror_descent.py::test_mirror_descent_scenario_1
tests/unit/optim/test_riemannian_adaptive.py::test_riemannian_contracts
tests/unit/solver/coupling/test_execution_contracts.py::test_members_with_different_custom_derivative_rules_never_share_lanes
tests/unit/solver/coupling/test_execution_contracts.py::test_members_with_different_host_callbacks_never_share_lanes
tests/integration/test_meshfree_bulk_surface_workflow.py::test_surface_source_program_rebinding_refuses_stale_reconstruction
```
