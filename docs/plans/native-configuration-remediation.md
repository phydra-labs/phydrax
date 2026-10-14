# Native configuration audit remediation

## Status and execution boundary

This plan addresses every finding in the configuration review: defects 1–13 and the aliased embedded-script enforcement gap. It has been implemented on the `native-configuration` worktree; section 13 records the outcome and deviations.

The reviewed source is the dirty `native-configuration` worktree. A later `//code--` continues there. A later `//code` must create a fresh worktree from that branch and transfer a consistent snapshot of its complete tracked and untracked implementation delta first; the branch HEAD alone does not contain the new substrate. Copy the authoritative ignored `AGENTS.md` before working. Do not implicitly commit, merge, discard concurrent changes, or edit the parent checkout.

**Only strictly affected tests may run. Never run the full suite, all integration tests, or an all-package test matrix.** This restriction supersedes the older global verification instructions in [native-configuration.md](native-configuration.md), including its full-suite command. During implementation, run exact failing nodes first, then the owning tests identified below; do not expand selection merely because a file imports the package root.

## 1. Contract decisions

1. Keep `resolve`, `configure`, `context`, `environment`, and `snapshot` as the single public operations. No compatibility aliases, extra resolver entry points, mutable wrappers, or executable configuration loaders.
2. Resolved input trees are recursively immutable and validated, including supplied exact dataclass instances. A class's custom attribute hook is not proof of a frozen declaration.
3. A dtype rule stores its declared NumPy dtype, not an import-time JAX canonicalization. JAX array contracts resolve that intent against the effective x64 setting when checked or converted; host array contracts retain the exact NumPy dtype.
4. Resolve and restore use one schema-aware canonical codec. Every union value has an explicit `case`/`value` envelope; nominal union alternatives declare their own stable kind. Normal Python/literal input remains separate from persistence syntax.
5. Enum records retain both canonical member name and canonical immutable value. Names alone cannot identify unchanged scientific intent when the member value changes.
6. Origins use `environment_keys: tuple[str, ...]`, replacing singular `environment_key` everywhere. Aggregate native inputs need all contributing keys, not an arbitrary representative key. Origins remain outside the value ID.
7. `_config_environment.py` owns the fixed native environment catalog and aggregate decoding. `_jax_config.py` alone owns JAX/XLA spellings and their external formats. The generic resolver imports the native decoder function-locally, not through a new top-level cycle.
8. Generic resolution reads only supplied/declared inputs and has no runtime effects. Native startup adds the unspecified-x64 default and foreign-launcher autodetection separately. Launch-only intent must never be passed into in-process application.
9. Child projection treats bootstrap as one selection, preserves custom timeouts, and recognizes both native and legacy forced-CPU topology. No hybrid of old and new rank settings.
10. Successful runtime effects and their recorded request are committed together. Later failure does not roll them back and must not record the failing field.
11. Package-attribute resolution protects the entire owner import and attribute lookup under process numerical settings, including default-device placement. Ordinary explicit imports inside numerical scopes remain the documented unsupported case; no import hooks or numerical-kernel rewrites.
12. Enforcement statically analyzes imports/aliases in Python files and extracted embedded Python material. It never evaluates source or imports targets named by a script.

## 2. Finding-to-change map

| Finding | Owning changes | Consumer-visible acceptance |
| --- | --- | --- |
| 1. Unchecked nested instances | `_config_resolution.py`: `_coerce`, `_record_items`, `_resolved_record`, construction/encoding validation | Invalid nested scalar kinds refuse; caller mapping mutation cannot change the resolved value, serialized value, or ID. |
| 2. Stale JAX dtype rules | `_dtype_names.py`, `_typing_forms.py`, `_typing_plan.py`, `typing.py` | Startup-off then process-on converts `Float64` to float64; temporary off scopes convert/check consistently and restore; compiled checks do not retain obsolete effective dtypes. |
| 3. Host forms narrowed by JAX | Same dtype owners | `HostFloat64`, `HostInt64`, and `HostComplex128` retain their NumPy widths with x64 off; host float32 does not satisfy `HostFloat64`. |
| 4. Lossy union/enum records | `_config_resolution.py`: schema compilation, canonical encode/decode, IDs; config fixtures/docs | Record alternatives with equal layouts have different IDs; exact selected types restore; enum/string alternatives and tuple-valued enums round-trip. |
| 5. Nested facade scope escape | `_deferred_exports.py` and manual lazy facade entry points | First access through root re-export or qualified nested facade cannot permanently narrow process-global numerical constants. |
| 6. Missing native environment layer | `_config_environment.py`, `_jax_config.py`, `_config_resolution.py` | Runtime/provider/native numerical and launch records resolve actual catalog inputs with correct precedence and truthful source-key tuples. |
| 7. Incorrect frozen check | `_config_resolution.py`: `_record` | Non-frozen dataclasses refuse even with a custom `__setattr__`; ordinary frozen definitions remain accepted. |
| 8. Stale optional bootstrap keys | `_config_environment.py`, `_execution_launch.py` | Override clears omitted local-device IDs and bind address through both public projection and rank launch. |
| 9. Dropped bootstrap timeouts | `_config_environment.py` catalog/projection/startup | Nondefault heartbeat/init timeouts survive projection and real child coordination unchanged. |
| 10. Defaults topology alias | `_jax_config.py` decoding/projection/observation | Existing forced-host topology survives defaults mode; unrelated flags do not suppress a new default; explicit override removes only the conflicting topology alias. |
| 11. Stale partial request | `_config_state.py` application/commit phases | Failed bootstrap leaves successful prefix visible in effective state and recorded request, with accurate `ConfigurationApplicationError.applied`. |
| 12. Invalid restored provenance | `_config_resolution.py`, environment catalog provenance validation | Exact terminal-path coverage, order, uniqueness, layer/key relationships, and declared source combinations are enforced. |
| 13. Caller device placement leaks | `_jax_config.py`: `process_numerics` | First access under device 1 does not force a later device-0 consumer to transfer constants when transfers are disallowed. |
| Enforcement gap | `tools/check_import_boundaries.py`, existing audit tests, applicable Ruff bans | Aliased access in a worker string or Python fence is detected with the owning file/line; shadowed non-JAX names are not mistaken for JAX. |

## 3. Resolution, identity, and provenance slice

### 3.1 `phydrax/_config_resolution.py`: immutable input preparation

- Replace the `__setattr__` heuristic in `_record` with authoritative dataclass frozen metadata. Keep cycle detection, runtime annotation resolution, public constructor-field validation, and refusal of derived/private override paths.
- Remove `_coerce`'s unchecked exact-instance return. Snapshot declared constructor fields through `_record_items`; recursively coerce them and construct through the trusted owning class, as for mapping inputs. This covers records under optional/union fields, tuples, and mappings.
- Full instances supply all constructor fields. Do not invoke unused default factories merely to reconstruct a complete instance. Partial mappings still obtain declared defaults and required-field errors through the existing preparation/assembly phases.
- Normalize mappings to the existing `frozendict` and sequence inputs to tuples only where the annotation permits tuples. Never mutate the original instance or caller mapping, bypass a constructor, or put a tuple into a list-annotated field.
- The canonical encoder also validates the constructed stored tree without conversion or callback invocation. Refuse a constructor/post-init hook that reintroduces mutable containers or violates the compiled field forms; do not silently repair the resulting object.
- Keep scalar rules: exact bool, no bool-as-int, no lossy integer conversion, finite real values, explicit selector parsing, bounded collections, and complete path diagnostics.

### 3.2 Canonical codec and nominal alternatives

Add private schema-owned encoding/decoding helpers in this existing file. Split only by real responsibilities: immutable preparation, schema/case compilation, value encoding, strict wire decoding, and provenance validation.

Compile each union alternative with its immutable case descriptor. Scalar cases are `bool`, `int`, `float`, `str`, and `none`. Literal/container cases use a deterministic description of the declared form, retaining literal member kinds and fixed/variadic tuple structure. Do not derive a case from runtime layout, union position, or Python module/class display names.

Every nominal record or enum used as a union alternative must directly declare a canonical `__config_kind__: ClassVar[str]`, including a sole nominal alternative in an optional field and any nominal form needed to distinguish a container alternative's case descriptor. Use namespaced cases `record:<kind>` and `enum:<kind>`. Reject missing/inherited/duplicate nominal identities before resolving values. Existing native `RuntimeBootstrap` and `LaunchEnvironmentConfig` already declare their kinds; update the optional `Solver` and nominal-alternative test definitions that currently do not.

Every union value, including `None`, uses exactly:

```json
{"case": "record:example.left", "value": {"count": 3}}
```

An enum/string union selects different cases. A union's string alternative does not become an enum during restore. Container alternatives place their data inside `value`, so user mappings containing `case` and `value` keys do not collide with the outer envelope. No untagged fallback, unknown case, extra envelope fields, or trial-and-error restore.

For a standalone or union-selected enum, canonical member payload is exactly:

```json
{"name": "PAIR", "value": [1, 2]}
```

The schema owns the enum class. Decode through its existing member table, require `member.name` to equal the supplied canonical name, and verify the encoded declared member value exactly. Refuse unknown names and alias spellings. Member values may be recursively immutable bounded tuples of `None`, exact builtin scalar data, and finite floats; reject mutable values, arrays, callbacks, and arbitrary objects at schema compilation. JSON list representation of a tuple is acceptable because decoding selects the trusted enum member rather than comparing a list with its tuple-valued `.value`.

Including both name and value makes a changed member value change the configuration ID. This is not a class import/target factory: wire case strings only index the alternatives already compiled from the caller-supplied trusted type. Union declaration order does not select identity; preserve existing unambiguous exact-type input preference and refuse values whose normalized runtime type/content still strictly matches multiple alternatives.

### 3.3 Restore versus ordinary input

- `resolve()` continues accepting typed instances, mappings, and data-only override literals with defaults < environment < values < overrides.
- Ordinary inputs do not acquire a second discriminator syntax. Explicit typed instances select ambiguous nominal alternatives; an ambiguous bare mapping continues to refuse.
- `ResolvedConfig.from_record()` must not feed wire values back into `resolve()`. Compile the trusted schema, require exact complete wire fields, decode that schema directly without consulting the environment or filling missing persisted fields, re-encode, and compare canonical wire content before verifying the ID.
- The encoder observes already-canonical values. It must not invoke coercion, constructors, factories, or callbacks to select an alternative. Reject zero/multiple strict matches rather than guess.
- Preserve tuple ordering. Canonicalize mapping/record JSON keys with the existing `canonical_json`/`canonical_fingerprint`; never use `repr` or pickle. Apply existing item/depth bounds and scalar refusal rules during wire decoding too.
- `ResolvedConfig.to_record()` verifies declared kind, immutable/schema-valid stored value, current ID/content agreement, and origin validity before emitting. A malformed manually constructed result must not emit a successful-looking record.

### 3.4 Snapshot encoding remains separate

Rename the old one-way `encode_value` projection to `encode_snapshot_value`, update `_config_state.ConfigurationSnapshot.to_record`, and remove the obsolete private spelling/export. It remains an observational JSON-compatible projection. Do not inject union envelopes into snapshots or pretend snapshots are restored by the configuration codec.

Resolved value IDs are computed exclusively from the schema-aware canonical encoding and the declared root kind. Do not add a second hash/identity substrate.

### 3.5 Plural origins and strict validation

Cleanly replace `FieldOrigin.environment_key` with `environment_keys: tuple[str, ...] = ()` and make `_Assignment` use the same tuple. No alias or fallback decoder for the old field.

- Runtime source keys are canonical, sorted, unique strings.
- The tuple is nonempty exactly for layer `environment`; other layers require `()`.
- Generic declared bindings produce singleton tuples. Native aggregate decoders report every contributing key.
- `to_record()` emits `environment_keys` as a JSON list. Restore requires the current exact origin fields and canonical list order.
- Enumerate terminal paths from the compiled schema: recurse through exact `_Record` forms only. Optional/union, mapping, tuple, enum, literal, and scalar fields remain terminal.
- Restored origins must cover that terminal-path set exactly once and in sorted order. Refuse unknown/missing/duplicate paths, record-parent substitutes, and child paths under terminal fields.
- Validate environment source tuples against the declared generic binding or the native catalog owner. Provider aggregate keys must match the present environment-sourced mapping entries. Bootstrap/launch aggregates need owner-defined required groupings and allowed optional subsets; their typed values cannot prove whether a default-equivalent optional input was absent or explicitly present.
- A source key can legitimately appear on several origin records. For example, `XLA_FLAGS` can source both forced CPU count and opaque launch flags. Uniqueness is per source tuple, not global.
- Expand a record-level value/override assignment into the correct descendant origins; only supplied descendants take that layer. Higher-precedence values clear the superseded environment-key tuple.
- Reject a `default` origin on a required field without a default-capable ancestor. Do not attempt to authenticate historical input values from source-key names alone: this is structural provenance validation, not a cryptographic attestation.

## 4. Dtype and runtime-contract slice

### 4.1 `phydrax/_dtype_names.py`

Make `exact_dtype_rule()` store the declared `np.dtype`, independent of x64. Keep `DTypeRule` immutable, preserve its exact/category invariant, and retain the existing category matching and scalar precision utilities. Do not alter `canonical_dtype`, scientific precision-name APIs, or unrelated consumers of those helpers.

### 4.2 `phydrax/_typing_forms.py`

Build both JAX and host registrations from declared rules. Do not snapshot effective x64 in `TENSOR_FORMS`; do not rebuild or mutate this global table when configuration changes. Keep alias identity, annotation names, dimensions, field order, and static base types unchanged.

### 4.3 `phydrax/_typing_plan.py`

Add one owning `effective_dtype(ArrayContract)` operation:

- JAX backend: canonicalize the declared exact dtype under the effective JAX x64 setting.
- Host backend: use the declared exact NumPy dtype unchanged.
- Category/unrestricted form: no exact target.

Use it in `_check_array` for matching and diagnostic dtype labels. Retain `_dtype_names.dtype_matches` for category matching and explicitly resolved exact rules where applicable. Keep parse/class/signature plan caches about declaration intent only; no effective dtype, bound scope, arrays, or transient owner is cached.

The lookup is host metadata: no array allocation, device discovery, synchronization, or new JAX equation. Calls inside compiled functions read it during tracing; existing JAX compilation context separates x64 routes. Do not modify checked wrappers or derivative rules merely to refresh caches.

### 4.4 `phydrax/typing.py`

Use the same effective target in `_target_dtype` for casting admission and conversion. Keep one-time conversion, explicit host/device boundaries, category dtype preservation, tracer refusal for host conversion, and caller-owned array reuse. Safe casting must still refuse a narrowing conversion; do not silently convert a restored artifact's historical arrays to a new width.

### 4.5 Documentation and identity

Update the typing guide/API distinction: JAX exact forms follow JAX's effective canonical width; NumPy host forms retain exact width. Scientific owners can require x64/precision independently and must preserve their own refusal/evidence.

Update the obsolete root comment claiming typing freezes canonical dtypes at import; keep the early startup ordering itself. Default-x64 scientific values, array leaves, nominal dimensions, static/dynamic partitions, RNG addressing, and provider records are not redesigned.

## 5. Deferred resolution and placement slice

### 5.1 `phydrax/_deferred_exports.py`

Protect the complete first-use resolution transaction: owner import, nested `getattr`, fallback resolution, and successful namespace publication. Do not exit `process_numerics()` between import and attribute lookup. Leave the cache untouched on failure; restore caller scopes on all exits. Already-loaded exports still use ordinary module attribute access without a new hot-path wrapper.

Keep `load_module()` and `load_all()` as canonical host loading operations. Preserve module ownership, export names, fixed registration lists, and optional-provider checks.

### 5.2 `phydrax/_jax_config.py`

Extend `process_numerics()` to enter the process-global default-device state, including `None`, alongside x64, matmul precision, promotion, and host-transfer state. Read that state through the adapter; do not query devices to invent a default. Restore the caller's live-device scope after success or failure.

Do not add a global device field to a configuration record, hide arrays in a static closure, or introduce a lock/race redesign. The excluded concurrent-load observation did not establish another bug.

### 5.3 Remaining manual lazy entry points

Apply the same shared context inside the existing host-only module getters, covering owner import and final lookup. Preserve their signatures, search/dispatch order, caches, provider guards, and export vocabulary. Prefer a local `with process_numerics()` envelope over adding `__wrapped__` chains to identity-sensitive objects. Do not wrap numerical callables returned by the getter.

The mapped manual owners are:

- `phydrax/execution.py`, `phydrax/precision.py`;
- `phydrax/acoustics/__init__.py`, `algebraic/__init__.py`, `backends/__init__.py`, `enforcement/__init__.py`, `exterior/__init__.py`, `ml/__init__.py`, `sampling/__init__.py`, `tensor_network/__init__.py`;
- `phydrax/applications/lattice_field/__init__.py`, `applications/nuclear_fuel/__init__.py`, `applications/power/__init__.py`, `applications/vortex_flow/__init__.py`;
- `phydrax/atomistic/interchange/__init__.py`;
- `phydrax/chemistry/__init__.py`, `chemistry/periodic/__init__.py`;
- `phydrax/discretization/discrete_velocity/__init__.py`, `discretization/meshfree/__init__.py`, `discretization/vortex/__init__.py`;
- `phydrax/geometry/design/__init__.py`, `geometry/surface/__init__.py`;
- `phydrax/interchange/fmi/__init__.py`;
- `phydrax/operators/integral/layer_potential/__init__.py`, `operators/integral/vortex/__init__.py`, `operators/path_integral/__init__.py`, `operators/quantum/__init__.py`;
- `phydrax/optics/__init__.py`, `optics/transport/__init__.py`;
- `phydrax/solver/__init__.py`, `solver/coupling/__init__.py`.

Reconfirm current entrypoint ownership before editing because the LSP workspace index did not consistently follow this dirty worktree. Facades already using `deferred_exports` need only the shared fix; their fallback bodies do not require a duplicate scope. Runtime foreign-object proxies such as the SGP4 attribute proxy are not package loaders and are excluded.

No class move, scientific constant rewrite, import hook, or unrelated eager-to-lazy conversion is planned. Keep explicit imports outside numerical scopes in documented usage.

## 6. Environment, launch, and partial-commit slice

### 6.1 `phydrax/_config_environment.py`: fixed native decoder catalog

Add a frozen internal decoded-input record containing relative terminal path, validated host value, and contributing source-key tuple. A non-extensible exact-type catalog covers `RuntimeConfig`, `RuntimeBootstrap`, `LaunchEnvironmentConfig`, `NumericsConfig`, and `ProviderConfig`; `LoggingConfig` has no ambient key and keeps its required explicit input.

Reuse actual constructor validation rather than inventing a second runtime checker. Emit only present inputs, preserving provider empty text. The resolver function-locally calls the catalog while visiting exact record nodes, prefixes paths for composed native records, and creates ordinary environment assignments. Optional/union/container paths stay terminal; do not silently activate an absent optional parent or make its child paths CLI-addressable.

Reject active duplicate/overlapping environment assignments from native decoding and user-declared bindings rather than selecting by iteration order. User definitions may explicitly bind any canonical external key; foreign launcher spellings are excluded only from implicit native catalog inputs, not globally banned for user declarations.

Provide the catalog-owned validation of persisted source-key combinations used by the codec slice. Never sample or persist the complete process environment.

### 6.2 `_jax_config.py`: upstream formats and topology aliases

Own JAX numerical/runtime/launch environment spellings and pure decoding of the supplied mapping: x64 and JIT booleans, platform priority and legacy platform alias, CPU count and forced-host alias, numerical selector strings, matmul precision, opaque XLA flags, and preallocation. For x64/JIT, the installed JAX configuration boolean owner accepts case-insensitive `y/yes/t/true/on/1` and `n/no/f/false/off/0`, without silently trimming extra whitespace; mirror that policy for a supplied mapping, not by temporarily modifying `os.environ`. Preallocation has a different installed GPU-plugin contract: absent/empty is unspecified; exact `false`, `False`, and `0` are false; other nonempty text is true. Do not incorrectly reuse the configuration boolean parser there. Native record validation still enforces positive CPU counts and supported selector/value contracts. Preserve upstream automatic/sentinel semantics as representable sparse intent, and explicitly refuse unrepresentable values rather than silently reinterpret them. Do not change process globals or call device accessors to parse an environment.

Keep all JAX key strings out of the generic resolver and Phydrax catalog. Normalize once, then let native record owners validate the resulting values.

Replace the cross-module tuple-of-selection/value dictionaries with an adapter-owned projection operation that modifies only the fresh result dictionary. Remove `environment_assignments`/`EnvironmentAssignment` after their callers migrate; no shim. Upstream-specific alias handling stays inside this operation.

- Defaults examines the original base for either native CPU count or a standalone forced-host-count token; unrelated `XLA_FLAGS` is not topology selection.
- Override writes canonical `JAX_NUM_CPU_DEVICES` and removes only conflicting forced-host tokens, preserving other flags' spelling, quoting, and order.
- If the base has both spellings, preserve upstream precedence. Decode/observe/project through one token interpretation rather than three regular expressions.
- Match token boundaries and quoted/escaped spans, not arbitrary substrings within another flag value. Confirm upstream duplicate-token behavior in the narrow owning smoke instead of guessing which count wins.
- When an explicit native count and the request's opaque flags both contain topology, the explicit native count is authoritative; document that known-selector normalization.

### 6.3 Bootstrap selection and timeouts

Add these owned keys in `_config_environment.py`:

- `PHYDRAX_HEARTBEAT_TIMEOUT_SECONDS`;
- `PHYDRAX_INITIALIZATION_TIMEOUT_SECONDS`.

Decode positive decimal integers into the existing `RuntimeBootstrap` fields. Keep their current field defaults and preserve ordered local-device IDs. The explicit coordinator/count/process-ID trio stays all-or-none. Supplemental local IDs, bind, and timeout inputs do not themselves request coordination; startup can combine them with foreign-launcher autodetection.

Projection for an explicit bootstrap emits the mandatory trio and both timeout values. Omitted local-device IDs and bind address are deletion entries, not omitted dictionary entries.

In override mode replace the complete group. In defaults mode any preexisting group selection preserves the entire base group; do not create a hybrid rank or silently repair a partial externally selected group. The owning startup boundary still reports malformed required combinations.

`startup_runtime()` reuses the pure decoders, adds x64=true only if no startup choice is specified, and augments foreign-launcher detection only at startup. Preserve explicit Phydrax selection priority. Exclude launch-only intent from the request passed to running-process `configure()`; root/config observation stays backend-inert.

### 6.4 `phydrax/_execution_launch.py`

Make `RankEnvironment.apply()` construct its existing bootstrap/runtime request and use `runtime_environment(..., mode="override")`. Remove its separate dictionary update serializer and delete obsolete `bootstrap_environment` after the last caller moves.

Do not add timeout fields to `RankEnvironment` or `LocalProcessLaunchPlan`; callers needing custom timeouts already supply `RuntimeConfig(bootstrap=...)`. `RankEnvironment.apply()` samples/copies its base exactly once. Add catalog-owned `bootstrap_timeout_overrides(base)`, a frozen typed result with independent optional heartbeat/init updates, which parses only the two timeout variables and does not decode or validate an old coordinator trio. Construct the new rank bootstrap using each present timeout and the `RuntimeBootstrap` constructor default for each absent timeout, then invoke canonical override projection. Malformed present timeouts fail before projection. This preserves custom launch-base timeouts while the new rank owns/replaces coordinator/count/process ID and replaces or clears local-device/bind settings; canonical projection emits both final timeouts. No timeout spelling or numeric default is duplicated in the launcher.

`_execution_bootstrap.coordinate()` already forwards both timeout fields correctly and needs no numerical/coordination redesign. Export workers continue consuming `runtime_environment` without an API change. Scheduler `valueFrom` rank templates retain their catalog constants; do not replace runtime process-ID injection with a static rank dictionary.

### 6.5 `phydrax/_config_state.py`

Keep admission of the complete request before side effects and preserve effect order: collectives, platforms, CPU count, x64, bootstrap. For each explicit field, prepare its next immutable request state, perform the effect when needed, and publish only after successful return.

A field already effective can commit explicit requested intent without another upstream effect. Preserve the existing meaning of `error.applied` as successfully executed effects; do not falsely report a skipped setter as a new side effect. The failing/later fields remain uncommitted. A no-coordination bootstrap may be recorded without initializing distribution.

Keep the existing lock and exception/domain evidence; do not catch every exception, suppress failure, or promise rollback. On full success the merged request must match existing sparse-update semantics. On later failure, successful effects and their request entries remain observable and can be projected truthfully.

## 7. Static enforcement slice

### `tools/check_import_boundaries.py`

Keep the existing source roots, exact adapter exemption, historical-plan exclusion, deterministic diagnostic format, and public/private boundary checks. Replace alias-blind matching for complete Python material with AST-based import ownership analysis.

Separate helpers by actual invariants:

1. Extract Python files, Python/py Markdown fences (including executable fence tags), and constant embedded strings/static concatenations with their containing-file source locations.
2. Track imports, direct capability imports, simple alias assignments, lexical scopes, and shadowing. Recognize aliased JAX roots, directly imported config/context capabilities, and upstream private config-module access. Do not classify an unrelated object's `.config` attribute by spelling alone.
3. Walk accesses and imported capabilities against the one banned-capability catalog.
4. Emit sorted, deduplicated canonical capability diagnostics mapped to the original file/line, including escaped-newline literals.

Do not execute ASTs, callbacks, imports, or string formatting. For partial illustrative snippets that cannot parse, retain a conservative non-executing token/direct-spelling fallback rather than silently skipping suspicious material. Dynamic source generation beyond statically recoverable literals is not represented as a proven pass; document that static analysis boundary. Normal executable source remains covered by Ruff as well.

### `pyproject.toml`

Retain the existing public JAX configuration bans and exact `_jax_config.py` exemption. If private upstream config imports are not covered, extend the same banned capability policy narrowly to that owner; do not alter unrelated per-file exemptions or weaken the text/script audit.

## 8. Exact file ownership and integration order

| Owner/slice | Implementation files | Dependent files |
| --- | --- | --- |
| Resolution/codec integration owner | `_config_resolution.py`, config-facing metadata in `_config_types.py` | `_config_state.py` snapshot encoder import, config unit tests, typing fixtures, configuration docs |
| Dtype owner | `_dtype_names.py`, `_typing_forms.py`, `_typing_plan.py`, `typing.py` | dtype/typing unit tests, typing docs, root comment |
| Loader owner | `_deferred_exports.py`, manual facade getters listed in section 5 | shared `_jax_config.py` process-scope change, startup/geometry/registry consumer nodes |
| Environment/application owner | `_config_environment.py`, `_execution_launch.py`, `_config_state.py` | shared `_jax_config.py` decoding/projection; launch/runtime/config tests and docs |
| Enforcement owner | `tools/check_import_boundaries.py`, applicable `pyproject.toml` bans | `tests/integration/test_import_boundaries.py` |
| Integration/documentation owner | docs, changelog, generated public API, benchmark driver/current evidence | every merged implementation slice |

Give `_jax_config.py` one active owner because environment decoding/projection and process-numerics placement share it. Give `_config_resolution.py` and `_config_state.py` one active owner each. Parallelize the independent dtype, codec, and enforcement work after agreeing on the plural-origin/native-decoder interface. Loader migrations can use the fixed process-scope API without independently editing the adapter.

Implementation dependency order:

1. Freeze the codec/provenance/catalog interface and identity cutover.
2. Fix frozen/nested validation; implement lossless codec and provenance contracts together.
3. Implement declared/effective dtype separation; verify both startup directions and scoped restoration before touching consumers.
4. Integrate native decoder/projection and partial commits against the completed provenance interface.
5. Integrate nested/manual loader protection and device isolation through the owning adapter.
6. Integrate enforcement changes, update callers/docs, and regenerate only relevant public data after the final implementation change.
7. Run the strictly affected final matrix once; reuse prior valid evidence on unchanged inputs.

Agents skip mid-flight builds, formatters, linters, tests, and benchmarks. The integration owner schedules those after coherent slices. Never launch several unrestricted `pytest -n auto` jobs concurrently.

## 9. Permanent regression contracts and strictly affected selection

| Test file | Contracts to add/update | Selection rationale |
| --- | --- | --- |
| `tests/unit/test_configuration_resolution.py` | Frozen metadata refusal; typed optional/union/container records validated and detached; nested bool-as-int refusal; nominal-kind admission; native environment precedence/composition; plural origins; optional/mapping child overrides still refused | Owns resolver input/precedence behavior. |
| `tests/unit/test_configuration_records.py` | Exact selected union types and distinct IDs; enum/string and tuple enum round-trip; enum name/value mutation and malformed envelope refusal; mapping key collision protection; bounded decoding; complete origins and valid aggregate sources; unchanged snapshot JSON contract | Owns canonical wire, identity, and restore evidence. Move/extend existing round-trip/tamper coverage rather than duplicate it. |
| `tests/unit/test_configuration_runtime.py` | Both process x64 transition directions; numerical scope restoration; real partial application failure and request/effective/applied agreement; no-op explicit-field recording; existing late-admission refusal | Owns process/scoped configuration transitions. |
| `tests/unit/test_configuration_environment.py` | Stale rank options cleared; custom timeout child round-trip; malformed timeout refusal; atomic defaults; CPU alias precedence; unrelated/quoted flags retained; launch-base timeout composition | Owns child environment semantics. |
| `tests/integration/test_configuration_startup.py` | Backend-inert startup; actual native environment versus startup defaults; first-use nested facade precision and exception restoration; two-device default-placement isolation | Owns cold interpreter/load behavior; isolate first-use and topology in children. |
| `tests/unit/test_dtype_names.py` | Declared exact rules do not narrow at construction; category/precision utilities unchanged | Direct declaration helper contract. |
| `tests/unit/typing/test_forms.py` | Host exact widths under either x64 mode; effective JAX width validation; unrelated wrong dtype still refused | Direct structural array contract. |
| `tests/unit/typing/test_convert.py` | Cached form reused across both x64 directions; host widths invariant; safe casting refusal; device reuse/transfer boundaries; compiled conversion under effective x64 | Direct conversion behavior. |
| `tests/unit/typing/test_signature.py` | Cached checked boundary reused across x64 settings, including jit; correct refusal of unrelated dtype | Signature plans consume the changed array checker. |
| `tests/unit/typing/test_strict_integration.py` | Reused class plan under both x64 modes; field validation and constructor-bypassing restore preserve declared structure/array identity, without silently narrowing stored artifacts | Strict fields consume the same checker. |
| `tests/unit/test_distributed_execution_substrate.py` | Rank environment projection/launch nodes only, including retained custom timeout base and disjoint local IDs | Only changed launch boundary; do not rerun unrelated workset/solver scenarios. |
| `tests/integration/test_import_boundaries.py` | Aliased worker string/fence access, direct imports, static concatenation, partial snippet handling, lexical shadowing, exact exemption and original-line diagnostics | Consumer behavior of the audit tool, not source-text assertions about package wiring. |
| `tests/typing/cases/configuration.py`, `tests/typing/installed/configuration.py` | `FieldOrigin.environment_keys` has type `tuple[str, ...]`; generic `ResolvedConfig[T]` inference preserved; old singular API is not retained | Actual public metadata annotation cutover. |

First-use loader regressions must assert consumer-visible numerical results, not private `_GL_NODES` storage:

- Access `geometry.CSGContinuationPolicy` and the qualified `geometry.design.CSGContinuationPolicy` in x64-off scope, exit, then evaluate an ellipse boundary measure against the independent complete-elliptic-integral reference in float64. Preserve `EstimatedMass` evidence and choose a moderate aspect ratio/tolerance justified by the reference and quadrature accuracy; prove the former narrowed-constant fault fails the case.
- First access under device 1, then evaluate a public geometry boundary-measure consumer on device 0 with device-to-device transfers refused. Prove caller device/numerical scopes restore, including import/lookup failure.
- Keep the existing cold native model/operator restore and plugin registry consumer nodes reached through changed loader entry points. Add a throwaway per-changed-facade export smoke rather than a permanent all-module/source-wiring test or all scientific-family suite.

Additional existing selections are limited to the `geometry/test_csg_continuation.py`, `geometry/test_geometry_nd_closure.py`, `test_plugin_registry.py`, and cold artifact restore nodes actually reached by these edits. Determine exact node IDs from the current source, not lost caches or guessed IDs. Default-x64 scientific test families, native provider builds, full integration directories, unrelated dimensions-only tests, and eight-device campaigns are excluded unless implementation introduces a concrete additional delta requiring them.

Use deterministic parametrization with diagnostic IDs, existing subprocess helpers, and narrow line-local ty ignores for deliberate static misuse. No mock echoes, imported-name checks, callback counters, soft assertions, source copies, or tests that merely do not throw.

Retain one compact implementation evidence ledger and measure changed helper complexity before/after. New/materially changed ordinary helpers target cyclomatic complexity at most 15, cognitive complexity at most 20, and 120 NLOC; split by schema preparation, canonical codec, environment decoding/projection, application commit, and source extraction/analysis rather than trivial forwarding helpers.

## 10. Runtime smokes, gates, and performance evidence

### Runtime proof beyond tests

After formatting each complete slice:

1. Resolve a configuration containing two explicit nominal alternatives and a tuple-valued enum; serialize/restore it in a fresh interpreter; verify selected types, immutable content, distinct IDs, and malformed-record refusals.
2. In separate fresh processes, start x64 off/on, reuse already-prepared conversion/field/signature plans, change/scope x64, perform a real computation, and verify expected JAX width plus independent NumPy host width. Trace/compile the changed typed path and observe a real result; checker metadata must add no host transfer or numerical equation.
3. Project explicit nondefault bootstrap timeouts and start one real loopback-coordinated child; observe its requested settings and readiness, then shut down. Run the two-rank local launch smoke only for the changed rank projection, checking actual indices/count and a numerical collective.
4. Project a base forced-host topology under defaults and observe the expected device count in the child. Use dedicated topology subprocesses, never forced devices in shared pytest fixtures.
5. Run the public nested-loader precision/device scenarios above and one cold artifact restore reached by the loader delta.
6. Invoke the audit tool on a temporary file/fence containing an aliased forbidden access and on a shadowed non-JAX control; observe the exact meaningful diagnostics. No script is executed.

For partial application, use the already-observed real unsupported launcher-detection failure to exercise the public error path. Do not replace it with a mock setter echo. The regression must preserve a successful prefix and refuse the failed bootstrap, not merely suppress its error.

### Verification waves

- Iteration: exact failing/regression nodes only, then the corresponding owning files in section 9 with one `-n auto` invocation. Run provider/device-isolated smokes separately.
- Changed-file configured Ruff/format and static diagnostics precede identity-sensitive verification.
- Final integration: `python tools/check_typing.py check`, applicable selector and checked-boundary audits, the changed import-boundary audit, and public API consistency. Run installed typing for the supported interpreter because the public origin annotation changes; do not widen interpreter support as part of this remediation.
- Regenerate `docs/data/public_api.json` once after the final public contract changes. The export set should stay unchanged; do not add internal helpers to the public facade. Capability declarations are unchanged, so do not invent or regenerate scientific capability evidence.
- Executable docs: select only `test_docs_python_examples_run[docs/guides_configuration.md]` and the typing-guide node if its runnable blocks actually change. No entire docs-example matrix or full documentation build by default.
- No full-suite command, all-integration selection, repeated valid evidence, or attribution-only dev rerun is permitted.

### Benchmarks

Use the existing `benchmarks/configuration.py` for current cold-root/config imports, bounded resolver/record costs, and numerical-scope entry after the implementation is stable and no tests/builds/agents compete. Extend its scenario only where needed to measure the changed schema codec or first-use scope path; do not add speculative benchmark infrastructure.

Compare candidate host/trace preparation against the retained material baseline; report phases separately. For the typed compiled smoke, measure lowering, compilation, warm execution, compiler memory/code evidence, and logical retained leaves if the implementation changes the numerical graph or working set. Do not claim that a faster cold import proves numerical speed or device scalability.

Existing measured `configuration_baseline.json` and older source-qualified records remain historical. Never edit their IDs, timings, or identity fields to resemble the new codec. A refreshed current record must come from a real current run, not hand-edited old measurements. Preserve the material environment/lock identity without modifying the user's parent virtual environment.

## 11. Documentation and clean-cutover checklist

| File | Required update |
| --- | --- |
| `docs/guides_configuration.md` | Tagged canonical unions, direct nominal union kinds, immutable enum vocabulary and name/value payloads, nested-instance freezing, native environment composition, plural aggregate origins, timeout projection, default/override semantics, partial-effect evidence, and supported lazy loading scope boundaries. |
| `docs/api/configuration.md` | Current `FieldOrigin` field/record shape, native binding semantics, timeout names, errors, and preserved public facade operations. |
| `docs/guides_typing.md`, `docs/api/typing.md` | Distinguish declared dtype, effective JAX canonical width, and exact NumPy host width; explain cached-plan/scoped behavior and historical-array refusal without conversion. Add one self-contained `python executable` guide block demonstrating host float64 preservation and JAX canonical-width changes across a restored scope; existing ordinary Python fences are not executable-doc nodes. |
| `docs/api/execution.md` | Canonical rank projection, timeout base composition, and explicit device/runtime fact discovery. |
| `CHANGELOG.md` | Every corrected contract family; explicitly disclose canonical value/ID changes and singular-to-plural provenance cutover. Do not claim all tests or general import-scope isolation. |
| `docs/plans/native-configuration.md` | Point to this remediation and replace obsolete full-suite instructions/current dtype-freezing claims; historical implementation evidence remains historical. |
| `phydrax/config.py` | Keep exports unchanged; update explanatory docstrings only if needed for provenance/record semantics. |
| `benchmarks/configuration.py` and freshly measured current output | Consume corrected record behavior without compatibility branches; retain historical baseline evidence unchanged. |
| `docs/data/public_api.json` | Regenerate after final public annotation/documentation cutover; do not hand-edit. |

Remove singular `environment_key` callers, obsolete snapshot encoder spelling, obsolete bootstrap serializer, old projection abstractions, and any test depending on a lossy wire shape. Migrate every first-party live caller; preserve historical plans/measurement quotations as historical rather than inventing old-ID aliases. If an actual durable configuration consumer is found at implementation-time references, use the existing lifecycle migration/compatibility substrate with an explicit migration decision—never a hidden decoder shim or schema-generation suffix.

No `_identity.py` exception, fake historical fingerprint, new PyTree partition, re-export alias, custom derivative wrapper, or numerical fallback is part of the plan. Legitimate source-derived ID changes from modified helpers must be acknowledged, not suppressed.

## 12. Completion criteria

- All 14 rows in section 2 have direct consumer-visible evidence and no placeholder/fallback implementation.
- Exact supplied records are recursively validated, independent of caller mutation, and genuinely frozen.
- JAX dtype checks/conversions track effective x64 without changing declaration caches; NumPy forms remain exact.
- Typed alternatives and enums restore losslessly, behavior-distinct alternatives have distinct IDs, malformed/current-wire/provenance cases fail closed, and no data selects an imported constructor.
- Native resolution is pure, catalog-derived, precedence-correct, and composed-record-aware; aggregate provenance is complete without fabricated source history.
- Projection and startup preserve timeouts, clear stale rank options, and honor topology/defaults precedence through the single owners.
- Partial failure leaves accurate successful-effect and requested-state evidence without rollback fiction.
- Root, nested, and remaining manual lazy attribute routes use process numerical/default-device settings and restore caller scopes; explicit imports inside scopes are not advertised as supported.
- Alias enforcement covers the reported embedded/fenced-script escape with meaningful file/line diagnostics.
- Every affected caller, typing fixture, authored doc, live canonical record contract, and relevant generated artifact uses the cutover representation with no obsolete aliases.
- Only the strictly affected verification matrix runs; results and limits are recorded accurately, and historical evidence is not rewritten.

## 13. Implementation outcome and deviations

All 14 rows of section 2 are implemented with consumer-visible regressions. Deviations and additions:

- Native origin validation dispatches on exact record types and adapter-owned key catalogs (`RUNTIME_ENVIRONMENT_KEYS`, `NUMERICS_ENVIRONMENT_KEYS`, `LAUNCH_ENVIRONMENT_KEYS`), not on record display names.
- Generic resolution decodes a native bootstrap only when an explicit coordinator key is present; supplemental rank keys alone are startup-only inputs.
- Found during integration: a process-wide `enable_x64` request made inside a matching numerical scope was skipped because the comparison read the scoped value. Application now compares against the process value (`_jax_config.process_x64_enabled`), with a regression.
- The static audit uses lazy, iterative scope analysis gated on capability names, analyzes only text that mentions JAX, and treats embedded strings that merely name a capability (bare expressions) as data. The repository scan takes about 10 s.
- Loader regressions observe public results: an ellipse perimeter against `4 a E(3/4)` at `rtol=1e-9` (the 48-point rule is accurate to about `2e-10`; float32-staged constants lose about seven digits) and the absence of live arrays on a caller's device. Both fail with the fixes reverted.

