# Native configuration substrate: end-to-end implementation plan

## 0. Status, goal, and scope

This was prepared as an implementation plan through source reads, static searches, and parent-owned LSP references, before any program ran. The initial substrate has since been implemented; section 12 records that outcome. A subsequent review found correctness defects; [native-configuration-remediation.md](native-configuration-remediation.md) plans their fixes, which are implemented, and supersedes this file's remediation verification instructions.

Goal: make `phydrax.config` the canonical substrate for Phydrax configuration, including typed Python composition, overrides, process/library settings, effective configuration reads, scoped numerical settings, and configuration records. First-party code must not access upstream JAX configuration directly outside one narrowly owned adapter.

The change includes package code, callers, embedded worker scripts, test setup, tools, benchmarks, examples, executable documentation, typing fixtures, architecture enforcement, and generated public API data. It is not complete if only `phydrax/__init__.py` stops calling `jax.config.update`.

The plan follows current source rather than restoring a historical tree. Before implementation, refresh affected symbols and callers in the implementation worktree; concurrent changes belong to their authors. Do not fabricate a commit baseline from this source inspection. Establish the actual comparison point at implementation time: the latest merge into `dev`, as required by repository policy.

### Implementation workspace

- A later `//code` request must use a fresh worktree from the current branch under the repository-required `../phydra-labs/.worktrees/` location. A `//code--` request stays on the current worktree.
- Use a neutral branch/worktree name such as `native-configuration`.
- Copy the authoritative ignored `AGENTS.md` into the new worktree before implementation work.
- Do not merge or commit implicitly. Do not undo unrelated changes.

### Explicit non-goals

- Do not move every scientific `Policy`, `Plan`, model parameter, or service object into a process-global configuration object.
- Do not create a second numerical, array-shape, identity, artifact, migration, provider-registry, or execution substrate.
- Do not adopt an external configuration framework, pickle configuration persistence, arbitrary CLI evaluation, target-path object factories, configuration aliases, or mutable object wrappers.
- Do not monkeypatch JAX, intercept third-party configuration calls, or promise control over all applications sharing a Python process.
- Do not rewrite numerical kernels, import-time scientific constants, PyTree partitions, RNG addressing, or scientific record formats merely to make the configuration facade lazy.
- Do not rewrite historical benchmark evidence to make it look newly measured.

## 1. Binding architecture decisions

### 1.1 Follow the existing facade convention

Use one public `phydrax/config.py` facade with explicitly owned private implementation modules. This follows `execution.py` and `precision.py`; it avoids a second public namespace convention and a monolithic implementation file.

The configuration owner owns input mechanics and ambient settings. Scientific owners retain validation of scientific meaning, policy IDs, prepared plans, numerical state, status, resource evidence, and actual provider identity.

| Concern | Configuration ownership | Retained native ownership |
| --- | --- | --- |
| Runtime settings | Typed requests, environment interpretation, precedence, effective JAX reads, supported mutation deadlines | Execution coordination, runtime discovery, meshes, communicators, observed device facts |
| Scoped numerical settings | Entry/restoration of supported JAX settings | Scientific precision choices and acceptance/evidence |
| Library settings | Logging emission enablement; declared native-library and worker-selection overrides | Logging transport/sinks/context; native loading, executable lookup, worker sessions |
| Run recipes | Typed immutable input composition, required inputs, dotted overrides, resolved records | Model/solver construction, scientific checks, arrays and PRNG keys |
| Persistence | Canonical configuration payloads and content identity using existing utilities | Repository transactions, configuration lineage, migration/rollback, run admission |
| Service settings | A service may explicitly consume a resolved record/reference | Instance-local auth, quotas, repositories, schedulers, secrets, provider registrations |

### 1.2 Public operations

The public surface has distinct operations, not alternate spellings of the same operation:

| API | Contract |
| --- | --- |
| `resolve(config_type, *, values=None, overrides=(), environment=None)` | Prepare one supported typed configuration from declared defaults, explicitly declared environment bindings, a Python value layer, and literal dotted overrides. Return `ResolvedConfig[T]` with the original typed result, canonical content identity, and field origins. No backend discovery, numerical preparation, provider loading, or global application. |
| `configure(settings)` | Apply a supported native runtime/library request. Validate the complete request and phase constraints before side effects. Do not apply arbitrary application run records globally. |
| `context(settings)` | Apply supported lexical numerical settings, using upstream context semantics and exception-safe restoration. It does not configure backend topology or replace explicit scientific policies. |
| `environment(settings, *, base=None, mode="override")` | Produce the pre-interpreter environment projection of a native runtime request. `base=None` inherits the caller's current environment; an explicit mapping is copied. `mode="defaults"` preserves supplied selections, including equivalent external platform inputs; `mode="override"` applies explicit requested values. Never mutate the supplied mapping or the current process. This is a launch operation, not an alias for `configure`. |
| `snapshot()` | Capture effective supported settings and configuration lifecycle readiness without enumerating devices, loading providers, capturing the whole environment, or collecting telemetry. Requested values and observed effective values remain distinguishable. |
| `x64_enabled()` | Allocation-free effective boolean query used by admission checks, dtype selection, and resource accounting. It must see an active upstream/native context, not a cached startup value. |
| `logging_enabled()` | Read the single config-owned logging emission flag, without constructing a snapshot. |
| `default_prng_implementation()` | Read the effective upstream default for current provenance consumers. Do not introduce a global seed or silently change a supplied typed key's implementation. |

The exact generic signatures must be pinned with the repository's ty fixtures. Accept supported class forms through an owning schema validator; do not use `Any`, false casts, `_typeshed` private imports, or a wrapper that spoofs a different result type.

### 1.3 Native input records

Add host-only, immutable records:

- `RuntimeConfig`: sparse process-setting intent. Cover current x64 use, platform selection, CPU device count/collectives, distributed bootstrap, and the explicit pre-interpreter JAX/XLA environment controls used by existing launchers. Omitted fields mean no override, not a reset of unrelated settings.
- `LaunchEnvironmentConfig`: launch-only JAX/XLA inputs, separated from in-process settings because their application deadline and environment merge contract differ.
- `RuntimeBootstrap`: move the coordination input to the configuration owner. Preserve remaining constructor semantics, timeouts, explicit/autodetect exclusivity, and ordered local-device selection. Move CPU collectives to the single `RuntimeConfig.cpu_collectives_implementation` field, and remove the old environment classmethod in favor of native resolution. Fix constructor assertions with owning validation, not permissive fallbacks.
- `NumericsConfig`: optional x64, matmul precision, dtype promotion, rank promotion, directional transfer guard, and disable-JIT context inputs used by current callers. `None` means no override. An explicit matmul `"default"` remains an explicit policy choice.
- `LoggingConfig`: logging emission enablement only. Do not move live handlers or correlation context into the record.
- `ProviderConfig`: explicit overrides for declared Phydrax native-library/worker settings. Preserve absent versus present-empty values and do not resolve libraries/executables while constructing the record.
- `ResolvedConfig[T]`: typed resolved input plus canonical configuration ID and origins; no runtime arrays, callbacks, backend objects, scopes, or live resources.
- `ConfigurationSnapshot`: effective supported native settings, declared library inputs, and non-initializing lifecycle readiness. It is not `RuntimeInfo`, a hardware inventory, or `HostTelemetrySnapshot`.

Use ordinary frozen host records where no numerical PyTree behavior is intended. Any genuinely numerical module must follow `StrictModule`, field-contract, and fixed-role rules. No new static numerical storage is planned.

Closed Phydrax selectors are declared once with runtime-resolvable `Literal` aliases or enums and parsed through `phydrax.typing.parse`. Reuse existing scientific selectors without widening them. JAX-defined matmul algorithm names at the external adapter boundary may remain validated string inputs: use JAX's owning context validator rather than copying a second complete vendor enum table. `FunctionalMatmulPrecision` and the operator-specific accepted subset remain scientifically owned and unchanged.

#### Concrete fields and signatures

Use these request fields rather than inventing alternative spellings during implementation:

| Record | Fields |
| --- | --- |
| `RuntimeConfig` | `enable_x64: bool \| None`, `platforms: tuple[str, ...] \| None`, `cpu_device_count: int \| None`, `cpu_collectives_implementation: CPUCollectives \| None`, `bootstrap: RuntimeBootstrap \| None`, `launch_environment: LaunchEnvironmentConfig \| None` |
| `LaunchEnvironmentConfig` | `xla_flags: str \| None`, `preallocate: bool \| None`, `memory_fraction: float \| None`, `allocator: str \| None`, `compilation_cache_directory: str \| None`. These are external launch inputs, not a new in-process cache controller or resource budget. Require a finite declared numerical fraction without guessing a provider range; inherited textual environment forms and existing evidence remain unchanged. |
| `NumericsConfig` | `enable_x64: bool \| None`, `matmul_precision: str \| None`, `dtype_promotion: DTypePromotion \| None`, `rank_promotion: RankPromotion \| None`, `transfer_guard_device_to_host: TransferGuard \| None`, `disable_jit: bool \| None` |
| `LoggingConfig` | `enabled: bool`; native startup default is false |
| `ProviderConfig` | Immutable `library_paths: Mapping[LibrarySetting, str \| None]` and `workers: Mapping[WorkerSetting, str \| None]`. An absent slot inherits; explicit `None` masks ambient input and selects the owner's fallback; a present empty string preserves that owner's existing empty-value semantics. |
| `RuntimeBootstrap` | Retain coordinator address, process count/index, ordered local device IDs, detection method/autodetect, bind address and heartbeat/initialization timeouts. CPU collectives are a process setting owned only by `RuntimeConfig`, not a duplicated bootstrap field. |
| `ResolvedConfig[T]` | `value: T`, `configuration_id: str`, `origins: tuple[FieldOrigin, ...]`; the explicit configuration kind participates in the canonical record |
| `FieldOrigin` | Declared field path, owning input layer, and optional external environment key name; no raw command argument or ambient secret value |
| `ConfigurationSnapshot` | Effective runtime and numerical request-shaped values, native logging flag, declared provider overrides/known inputs, and non-initializing backend/distributed readiness. Automatic selection remains `None`; never discover hardware to replace it with a guessed value. |

Native selector owners declare `CPUCollectives`, `DTypePromotion`, `RankPromotion`, `TransferGuard`, `LibrarySetting`, `WorkerSetting`, and `EnvironmentMerge` once. `EnvironmentMerge` distinguishes `override` from `defaults` without creating a second environment builder. Platform identifiers are an open provider namespace, not a hardcoded CPU/GPU/TPU-only selector. Do not duplicate a scientific precision selector under a different name.

`configure` accepts `RuntimeConfig | LoggingConfig | ProviderConfig` and returns `None`. `context` accepts `NumericsConfig` and returns `ContextManager[None]`; it is not a decorator and does not wrap a scientific callable. `environment` accepts `RuntimeConfig` and returns `dict[str, str]`. `snapshot` returns `ConfigurationSnapshot`; the two boolean getters return `bool`, and the default-PRNG getter returns `str`.

Supported Python configuration definitions declare `__config_kind__: ClassVar[str]`. Optional environment bindings use `__config_environment__: ClassVar[Mapping[str, str]]`, mapping declared field paths to authoritative key names. Native definitions derive bindings from the one environment catalog; do not restate that catalog in each class. No abstract base, implicit class-name identity, or global registry of user config classes is required.

Dotted traversal reaches nested record fields and declared closed-key mapping slots. Unconstrained mappings are terminal data and are replaced atomically; do not reinterpret arbitrary dotted mapping keys as object paths.

Wrong record/scalar kinds raise `TypeError`; invalid values, unsupported schemas, override paths and missing inputs raise `ValueError`. Late backend/bootstrap application raises `RuntimeError` or a native application error derived from it with setting/phase/applied-field evidence. Existing scientific domain errors propagate unchanged. Never turn a failed application into a successful-looking snapshot.

Opaque external launch strings are transport inputs, not a credential store. Display/persist only explicitly classified nonsecret fields or existing reference contracts; do not assume arbitrary free text becomes safe merely because its key is allowlisted. Existing performance/environment evidence retains its owning privacy and identity contract.


### 1.4 Startup and import ordering

Make the root package device-inert, not artificially JAX-import-free:

1. Read and resolve declared startup environment inputs.
2. Apply the default/declared JAX settings through the adapter.
3. If requested, initialize distributed coordination without runtime/device discovery.
4. Return from root import with numerical subpackages still deferred.
5. Resolve a requested numerical module/symbol through a lazy root facade or a normal direct submodule import.
6. Numerical construction/execution may initialize devices only after configuration/bootstrap.

Both `import phydrax; phydrax.geometry` and `import phydrax.geometry` execute the root initializer first. This provides one startup boundary without import hooks, `sitecustomize`, or per-module activation prologues.

The root may import JAX and host metadata dependencies. It must not call `jax.devices`, `jax.local_devices`, `jax.process_count`, `runtime_info`, or numerical constructors to inspect readiness. Installed JAX's `process_count` traverses backend devices; it is not a harmless configuration query.

### 1.5 Default and lifecycle behavior

- Keep x64 enabled as the Phydrax default when no explicit startup request says otherwise.
- Respect an explicit startup `JAX_ENABLE_X64=0` instead of unconditionally overwriting it. This is an intentional behavior change, not an identity-preserving move of a line of code.
- Do not infer whether a preexisting JAX value came from an explicit API call or an upstream default; JAX does not provide that provenance. Define the native default/environment/explicit precedence rather than fabricating origins.
- Backend-bound changes such as topology, platform selection, and CPU collectives must reject changed values after backend initialization. Identical requests may be idempotent; never pretend a late changed request took effect.
- x64 and other supported trace/context settings retain their actual JAX mutation/context capabilities. Do not blanket-lock x64 merely because a backend exists. Document that a process change affects subsequent operations, not the stored dtype of existing arrays or historical execution.
- Environment-only client/XLA controls require the appropriate pre-interpreter/pre-consumption phase. When that phase has passed, refuse in-process application and direct the caller to `environment()` plus a fresh process. Do not silently edit `os.environ` and claim success.
- Resolve/validate before applying. Resolution is atomic; distributed initialization is not generally reversible. A failure must identify the setting/phase and any effects already applied. Do not advertise transactional rollback of native runtime initialization.
- Managed numerical contexts use JAX's thread-local semantics. Do not claim asynchronous task isolation or automatic propagation to new threads/processes. Logging correlation remains `ContextVar`-owned by logging.

### 1.6 Clean cutovers

Remove, rather than alias:

- `phydrax.execution.RuntimeBootstrap`; canonical public input becomes `phydrax.config.RuntimeBootstrap`.
- `RuntimeBootstrap.from_environment` and its old CPU-collectives constructor field; native resolution owns environment parsing, and `RuntimeConfig.cpu_collectives_implementation` is the single process-setting input. The external `PHYDRAX_CPU_COLLECTIVES` spelling stays unchanged.
- `phydrax.execution.initialize_from_environment` and its private implementation; startup configuration consumes the environment, while `execution.initialize()` remains the explicit runtime/discovery action.
- `phydrax.logging.enable`, `disable`, and `is_enabled`; callers use `configure(LoggingConfig(...))` and `logging_enabled()`.
- Every direct first-party upstream JAX configuration access outside `phydrax/_jax_config.py`.
- Superseded environment-name constants and duplicated host-setting lookup logic after their consumers are migrated.

Retain `execution.initialize`, `runtime_info`, `RuntimeInfo`, `RankEnvironment`, resource planning, logging sink operations, logging correlation, and scientific policy APIs. No forwarding compatibility methods or old re-exports.

## 2. Resolution, records, and environment contracts

### 2.1 Python-first configuration without mutable wrappers

Accept supported immutable typed dataclass records and nested host input records, not arbitrary mutable object graphs. No mandatory `Config` base class is needed.

Resolution must:

- Inspect declared constructor inputs; reject overrides of `init=False`/derived fields, methods, private attributes, IDs, and evidence fields.
- Resolve annotations in their owning module/class context, not only under `TYPE_CHECKING`.
- Support ordinary scalar data, declared optional/union/selector forms, typed tuples, supported immutable mappings, and nested supported records.
- Preserve declared types. A list literal may be normalized to a tuple only where the declared input is a tuple; do not put a tuple into a field truthfully annotated as `list` and call it immutable.
- Reuse the canonical immutable mapping substrate where appropriate; do not add another frozen-dictionary implementation. Numerical/native policy objects are constructed through their owners after startup, not mutated via `__dict__`, `tree_at`, or unchecked reconstruction.
- Reject arrays, tracers, legacy keys, callback-bearing objects, live devices, open handles, and arbitrary provider objects as host configuration values. Use existing explicit artifact/policy references at those boundaries.
- Treat native constructors and explicitly supplied Python configuration definitions as trusted host preparation. Do not execute targets, imports, methods, or callbacks selected by CLI/data text.
- Refuse unsupported placements rather than partly enforcing an annotation and dropping accepted branches.

### 2.2 Layers and overrides

Precedence is declared defaults, then declared environment bindings, then an explicit Python value layer, then CLI/programmatic literal overrides. Environment bindings are explicitly declared by the native record owner or supplied configuration definition; do not invent environment names from display strings.

Within an override layer, reject duplicate paths and parent/child overlap with a clear path-specific error. Independent overrides are order-independent. Across layers, higher-precedence values win. This avoids order-dependent mutation of already-constructed objects.

The literal grammar is data only: quoted strings, booleans, `None`, finite numbers, and bounded supported collections. Bare ordinary words can represent strings; executable expressions and method syntax are rejected. Bound input size, nesting, and collection capacity before recursive materialization. Narrow parsing exceptions are appropriate; a broad catch-and-return-string fallback is not.

Scalar policies must be explicit: never accept `bool` as an integer, truncate `5.2` into `5`, or coerce arbitrary values with `str`. Floating inputs may use the owning finite-real conversion where that is the declared contract. Scientific positivity, units, support, rank, and conditioning stay with the domain constructor.

### 2.3 Configuration identity and persistence

- Each supported configuration definition declares an explicit canonical configuration kind. Native kinds are owner-declared; application kinds are explicit metadata, not inferred from a class display name, equal dimensions, or array shape.
- A resolved configuration ID hashes its canonical declared kind and resolved value record through existing `canonical_fingerprint`. Field origins are separate provenance; equal canonical intent does not acquire a different scientific/configuration content ID merely because it came from a different input layer.
- Preserve order of semantically ordered tuples, including platform priority and local-device assignment. Sort unordered record keys and origin-path records canonically.
- Use `canonical_json`; retain `None`, booleans, numbers, tuples/records through explicit canonical encoding. No `repr`/string fallback, pickle/dill, executable YAML loader, generation suffix, or schema-version field.
- Decode with a caller-supplied supported type/kind; validate exact fields and the recomputed ID. Never import a class named by an untrusted serialized `_target_`.
- Bind native configuration records to `ResolvedRunSpec.prepared_configuration_id` where a qualified run needs that reference. Do not add another run-spec ID field, replace policy IDs, or change `numeric_revision_id`.
- Store/migrate records through current lifecycle repositories, `CompatibilityRegistry`, `migrate_configuration`, and rollback. No parallel archive or migration graph.
- Native process snapshots do not contain credentials, raw CLI strings, or the entire environment. Service secrets remain explicitly injected through existing service contracts such as `SecretHandle`. Redaction is not a substitute for excluding credential payloads from the native record contract.

### 2.4 Declared host-setting catalog

| External key(s) | Native use and preserved semantics |
| --- | --- |
| `JAX_ENABLE_X64` | Explicit startup x64 request; native default is true only when unspecified |
| `JAX_PLATFORMS`, current `JAX_PLATFORM_NAME` launch inputs | Platform intent; preserve current launch semantics and priority, and route new native selection through the current public JAX owner without silently changing old launch contracts |
| `JAX_NUM_CPU_DEVICES` / explicit forced-host `XLA_FLAGS` recipes | Distinct topology inputs; preserve forced-host functional versus physical campaign classification |
| `XLA_FLAGS`, preallocation/allocator/memory-fraction/cache environment controls used by existing launchers | Opaque external launch controls, phase-classified and projected before the child starts; not hardware facts or cache-byte budgets |
| `PHYDRAX_COORDINATOR_ADDRESS`, `PHYDRAX_NUM_PROCESSES`, `PHYDRAX_PROCESS_ID` | Required together; retain the current refusal contract |
| `PHYDRAX_LOCAL_DEVICE_IDS`, `PHYDRAX_COORDINATOR_BIND_ADDRESS`, `PHYDRAX_CPU_COLLECTIVES` | Optional bootstrap inputs; preserve value/blank/ordering behavior and closed collectives validation |
| `SLURM_NTASKS`, `OMPI_COMM_WORLD_SIZE`, `PMI_SIZE`, `PMIX_SIZE` | Foreign launcher discovery inputs; no renaming or fabricated Phydrax equivalents |
| `PHYDRAX_MESHCORE_LIBRARY`, `PHYDRAX_METIS_LIBRARY` | Lazy library selection only; existing loaders retain their distinct blank/fallback/error/ABI/cache behavior |
| `PHYDRAX_MMG_WORKER`, `PHYDRAX_OMEGA_H_WORKER`, `PHYDRAX_TIOGA_WORKER`, `PHYDRAX_VOROCRUST_WORKER` | Lazy worker selection; explicit constructor input wins, then native override/environment, then the owner's default. A present empty worker variable remains different from absence |
| `PHYDRAX_WORKER_MEMORY_LIMIT_BYTES` | Internal child transport written finally from `NativeWorkerPolicy`; not public ambient host configuration |
| `PHYDRAX_BENCHMARK_FORCED_HOST_DEVICES` | Benchmark IPC/evidence marker, not a public library setting |

Do not add qualification-only pinned-provider variables to the native runtime catalog merely because their spelling starts with `PHYDRAX_`. Likewise, `GEANT4_DATA_DIR`, `MPLBACKEND`, OpenFOAM variables, Torch worker security controls, CMake inputs, and request-specific external environment maps retain their external/worker/build owners.

Host library/worker lookup samples declared environment values at the existing lazy lookup point unless the caller supplies explicit native overrides. Do not freeze all provider selection during package import. A configuration path is not the identity of the library or worker actually loaded.

## 3. Core implementation files

| File | Exact responsibility/change | Acceptance |
| --- | --- | --- |
| **New `phydrax/config.py`** | Explicit public facade for the records and operations in section 1; static literal exports, truthful typing, no numerical-package import for configuration access | Import/configuration access is device-inert; supported public symbols resolve without compatibility aliases |
| **New `phydrax/_config_types.py`** | Immutable host requests/results, declared configuration kinds, origins, phase classifications, native closed selectors; move `RuntimeBootstrap` here | Constructor inputs validated before canonical storage; no arrays, scopes, resources, or static checker state in records |
| **New `phydrax/_config_resolution.py`** | Owning typed input compiler/resolver, bounded literal parsing, layered overrides, required inputs, atomic nested preparation | Unknown/derived paths, duplicate/overlap, missing inputs, invalid scalar types, unsupported forms, and expressions fail before application |
| **New `phydrax/_config_environment.py`** | One declared external-key catalog, absent/empty-aware lookups, typed startup decoding, pure native environment projection | Supplied mappings unchanged; only owned keys interpreted; lazy provider reads and child environment contracts preserved |
| **New `phydrax/_config_records.py`** | Explicit canonical value encoding, configuration content ID, origins, snapshot construction/strict decoding using current fingerprint/privacy utilities | Deterministic records, lossless supported values, tamper/unknown-field refusal, no whole-environment capture or repr fallback |
| **New `phydrax/_config_state.py`** | Single host-only applied library/startup state and logging enablement; validated commit ordering and explicit failure phase; no hidden run recipes | No duplicated flag, no numerical leaves/global transient owners; concurrent application cannot silently interleave a compound request |
| **New `phydrax/_jax_config.py`** | Sole upstream JAX configuration adapter: public setters/context managers/effective reads, adapter-side phase validation, narrow backend-readiness query, JAX/XLA projection | No first-party bypass; native contexts restore exactly; no backend initialization caused by a readiness check |
| `phydrax/__init__.py` | Replace eager runtime imports with configuration startup plus an explicit PEP 562 lazy module/symbol map; retain precise `TYPE_CHECKING` imports, current declared public exports, and add `config` | Direct submodule and lazy-root paths establish startup first; root import/config snapshot causes no device work; existing root symbols retain their canonical owners |
| `phydrax/_execution_bootstrap.py` | Remove configuration type/environment constants and direct JAX config write; retain `RuntimeInfo`, discovery, and explicit execution initialization. Separate coordination-only initialization from runtime discovery | Import-time coordination never calls `runtime_info`; actual `initialize()` still returns observed runtime facts and retains late/double-init refusal |
| `phydrax/execution.py` | Consume configuration-owned `RuntimeBootstrap`; remove its old export and `initialize_from_environment`; retain execution actions/facts | Current execution planning API remains; removed paths have no shims |
| `phydrax/_execution_launch.py` | Use canonical owned rank-key serialization while retaining `RankEnvironment`, shell-free launch, inherited base environment and optional-key behavior | Same rank ordering, overlay precedence, local IDs/bind-address handling, return codes, and failures |
| `phydrax/interchange/__init__.py` | Provide lazy `cosmology` attribute resolution after removing root's unconditional cosmology interchange import | Existing `phydrax.interchange.cosmology` access works without startup importing applications |
| `phydrax/_model/_artifacts.py` | Retain the existing registry/IDs. Add owner-controlled loading of known built-in registration families before a cold lookup; unknown/custom IDs still require explicit registration | No arbitrary module imports from artifact text, display-name lookup, wildcard plugin loading, second registry, or ID changes |
| `phydrax/_model/_structure.py` | Keep complete reconstruction validation and exact type/field checks; consume the same canonical registry after built-in owner preparation | Portable arrays/types/keys reconstruct in a cold process with unchanged exact inventories and identities |
| `phydrax/nn/operator/training/_artifact.py` | Ensure its canonical architecture/recipe owners are ready before codec/template lookup | Template-free portable restore works without root eager preload; unknown architecture and malformed recipe refusal unchanged |
| `phydrax/atomistic/_model_artifact.py` | Reuse its existing `_register_native_models()` restore preparation; verify it does not depend on root imports | Native model cold restore still works without importing an external provider |

### JAX private dependency exception

The installed supported JAX source has no general public non-initializing backend-readiness query. `jax.distributed.is_initialized()` is insufficient, and public backend enumeration initializes backends. Encapsulate `jax._src.xla_bridge.backends_are_initialized()` only in `_jax_config.py`, with a concise ownership rationale and dependency contract coverage for the supported JAX range. Use public JAX setters/contexts for everything else. If the required dependency contract is unavailable, fail explicitly; do not probe devices or add a permissive fallback.

### Built-in artifact owner preparation

Root laziness must not depend on incidental prior imports to populate the existing registry. Resolve only fixed built-in registration families to their canonical owner modules, then perform the existing exact registry lookup. Do not derive an import path from the serialized value ID.

Registration-bearing source files to audit for this preparation, with their existing registration bodies/IDs otherwise retained:

- `phydrax/nn/_utils.py`
- `phydrax/nn/layers/__init__.py`, `phydrax/nn/models/__init__.py`, `phydrax/nn/parameters/__init__.py`
- `phydrax/nn/atomistic/_mace.py`, `_mace_interaction.py`, `_mace_prepare.py`, `_mace_readout.py`, `_radial.py`, `_radial_projection.py`, `_symmetric_contraction.py`
- `phydrax/nn/operator/adapters/__init__.py`, `phydrax/nn/operator/architectures/__init__.py`
- `phydrax/nn/operator/architectures/conditioning/_deeponet.py`, `_function_frame.py`
- `phydrax/nn/operator/architectures/dynamics/_conditional_affine.py`
- `phydrax/nn/operator/architectures/spectral/_cno.py`, `_fno.py`, `_sfno.py`, `_wavelet.py`
- `phydrax/nn/operator/layers/_o3.py`, `_o3_tensor_product.py`, `_regional_processor.py`
- `phydrax/nn/operator/representations/_irreps.py`, `_o3.py`
- `phydrax/nn/operator/training/_physics.py`

Do not modify these numerical owners simply to restyle imports. Their fixed registration families and actual lookup consumers determine whether a source change is needed. Keep the interpolation `UniformNodeGrid` and polynomial `ProductGraphPlan` registrations currently supplied by their first artifact consumers; do not create an import cycle by moving them blindly.

## 4. Package JAX caller cutover

For every row, replace only the upstream configuration access. Preserve the existing scientific predicate, error type/message/order, dtype/index calculation, evidence record fields, and numerical operation order. Do not cache effective settings at module import.

| File | Required migration |
| --- | --- |
| `phydrax/_bvh.py` | Native x64 query for the int64 overlap-pair admission gate |
| `phydrax/_interpolation/_rectilinear.py` | Native x64 query for index dtype and resource-byte calculation |
| `phydrax/applications/electrophysiology/_events.py` | Native query for event sequence-counter dtype |
| `phydrax/applications/geophysics/_vertical.py` | Preserve default dtype selection and explicit float64 refusal through native effective x64 |
| `phydrax/applications/robotics/_environment.py` | Native effective default-PRNG read; preserve its existing provenance field/value |
| `phydrax/applications/skeletal_muscle/continuum/_almonacid_2024.py` | Preserve source-matched float64 admission gate |
| `phydrax/discretization/explicit_polygon_h1/_precision.py` | Preserve declared-role float64 admission |
| `phydrax/discretization/fem/_precision.py` | Preserve declared-role float64 admission |
| `phydrax/discretization/finite_volume/_unstructured.py` | Preserve lossless/rejected stable-ID width and device global-ID dtype choice |
| `phydrax/discretization/meshfree/_neighbors.py` | Preserve uint64 Morton/int64 addressing requirement |
| `phydrax/discretization/meshfree/_precision.py` | Preserve native float64-role refusal |
| `phydrax/discretization/spatial/_distributed_relations.py` | Preserve distributed Morton/stable-ID requirement |
| `phydrax/discretization/spatial/_neighbor_query.py` | Preserve spatial address requirement and float32-coordinate distinction |
| `phydrax/export/_atomistic.py` | Native effective x64 read in runtime record and worker environment; use native launch projection while preserving archive keys, isolation allowlist, CPU target, `EQX_ON_ERROR`, parity and runtime mismatch refusal |
| `phydrax/export/_atomistic_export_worker.py` | Respect projected configuration before reconstruction/trace; retain runtime equality/refusal, exact source-root check, native status/parity, and atomic publication |
| `phydrax/finance/core/_currency.py` | Preserve exact int64 monetary-atom gate |
| `phydrax/nn/operator/training/_execution.py` | Native numerical context around prediction; preserve `None` versus explicit precision, typed model leaves and compiler entrypoint identity |
| `phydrax/optim/_programming/_exponential_cone.py` | Preserve effective working dtype/promotion using native query |
| `phydrax/optim/_programming/_power_cone.py` | Preserve effective working dtype/promotion using native query |
| `phydrax/service/_observability.py` | Native configuration read only; retain explicit telemetry policy and existing runtime/device discovery semantics |
| `phydrax/solver/_functional_gradient.py` | Native precision context and config-owned logging-enabled import; preserve objective/gradient paths, physical/surrogate partition, and diagnostics |
| `phydrax/solver/_functional_solver.py` | Native context for objective settling; keep explicit `FunctionalPrecisionPolicy` and its evidence |
| `phydrax/solver/_functional_evolution.py` | Move logging-enabled import to config owner; no evolution/RNG/selection changes |
| `phydrax/solver/_kfac_solver.py` | Move logging-enabled import to config owner; no optimizer/layout/state changes |
| `phydrax/solver/_projector_monte_carlo.py` | Native x64 query at both gates; preserve local-array/process admission, counters and rollback |
| `phydrax/solver/_projector_monte_carlo_contracts.py` | Native x64 query at construction; preserve scientific problem/plan checks |
| `phydrax/velocimetry/piv/_plan.py` | Preserve current requested/resolved precision and corresponding complex dtype; do not turn its existing fallback into a different scientific policy |

`phydrax/solver/_functional_precision.py` and `phydrax/nn/operator/training/_dtype.py` remain canonical scientific policy/selector owners. Their accepted subsets, field order, IDs, serialization and evidence are deliberately unchanged. No alias/re-export is added under config.

The false-x64 atomistic export replay risk found in source is an unexecuted inference: the parent projects false but current root forces true. Before changing replay behavior, exercise the narrow worker configuration path. Preserve the distinction between a supported float32 execution and a correctly refused float64-only scientific route; do not bypass the refusal or reinterpret the same archived model.

## 5. Non-JAX library, worker, service, and lifecycle files

| File | Required action/invariant |
| --- | --- |
| `phydrax/logging.py` | Remove `_ENABLED` and the three old configuration APIs; `emit` reads one config-owned boolean. Retain named Loguru enable/disable effects, default silence, context restoration, sink ownership, formatting/redaction and application-handler independence. Remove import-time reset that would overwrite an already applied native logging choice |
| `phydrax/_meshcore.py` | Query configured path through native host settings at `_current`; retain blank fallback, configured-invalid refusal, package location, ABI/release checks and path-keyed cache |
| `phydrax/meshing/_metis.py` | Native configured-path query; retain blank/unset loader fallback, missing explicit path refusal, index-width evidence and cache |
| `phydrax/meshing/providers/_worker.py` | Native declared worker lookup at `_resolve`, not constructor time. Keep parent `PATH` search, explicit input precedence, empty-env semantics, sessions and relaunch |
| `phydrax/meshing/providers/_mmg.py` | Canonical worker-key selection; retain explicit executable contract and native worker policy |
| `phydrax/meshing/providers/_omega_h.py` | Canonical worker-key selection; retain explicit path and provider capabilities |
| `phydrax/meshing/providers/_tioga.py` | Canonical worker-key selection; retain `TiogaOptions`, launcher and child environment separation |
| `phydrax/meshing/providers/_vorocrust.py` | Canonical worker-key selection; retain explicit worker input and extraction behavior |
| `phydrax/_external_runtime.py` | Keep policy/security/environment builders owned here. Reuse canonical internal key spelling where needed, but preserve inheritance/allowlist validation, temporary-directory ordering, explicit overrides and final policy-derived worker memory limit. Do not replace both builders with one generic merge that changes precedence |
| `phydrax/service/_schedulers.py` | Use canonical rank-key vocabulary while keeping Kubernetes field references, scheduler-owned addresses, resource calculation and request-specific environment construction |
| `phydrax/service/_runtime.py`, `phydrax/service/_contracts.py` | Keep instance-local injection, registry, idempotency, submission digest and durable serialization. Native records may be referenced through the existing resolved-run contract; no global service settings or record-shape change |
| `phydrax/lifecycle/_resolved_run.py` | Existing `prepared_configuration_id` is sufficient; no new field or payload shape. Add integration coverage with a real native configuration record |
| `phydrax/lifecycle/_archive.py`, `phydrax/lifecycle/_migration.py`, `phydrax/lifecycle/__init__.py` | Keep migration graph, lineage transaction and rollback as canonical owners; consume current configuration records through existing APIs without adding a second persistence API |
| `native/providers/common/phydrax_worker.hpp`, `native/providers/mmg/worker.cpp` | Retain existing internal memory-limit transport spelling and enforcement. No native build change is planned because the wire contract is not changing |
| `phydrax/applications/detector/_geant4_providers.py` | Retain explicit external data-directory child contract; no host-global discovery or snapshot of datasets |
| `phydrax/interchange/_dafoam_worker.py` | Retain bounded external runtime evidence fields; these are observations, not native defaults |
| `phydrax/interchange/_hfss_design_worker.py` | Retain worker-local headless rendering setup and provider boundary |
| `phydrax/atomistic/interchange/_mace_worker.py` | Retain worker-local Torch security/loading controls; do not expose a public override that defeats the existing worker contract |
| `native/meshcore/python/phydrax_meshcore/__init__.py` | No direct configuration access found; retain its provider packaging API |

## 6. Frontend migration recipes and exceptions

Apply the same recipe to every file listed in appendix A. The inventory is a source snapshot, not permission to edit unseen source; refresh and use syntax-aware/LSP-supported edits during implementation.

### Recipe P: process x64 writes

Replace direct `jax.config.update`/imported `config.update` with the typed native runtime request, preserving the requested boolean and the original lifetime/timing. Do not turn a module-level process change into a context that exits immediately, or make a conditional float32/float64 request unconditional.

Remove a redundant setter only after proving its effect is already owned at every actual consumer boundary, including importers. In particular, do not simply delete `tools/nuclear_fuel_qualification.py`'s module-level x64 effect or move it to `main`; its importing tests, benchmark and examples must preserve their float64 requirements. This plan does not require a wholesale pytest-fixture or tool-import architecture refactor.

### Recipe R: effective reads

Use native effective getters. Preserve guards, skip decisions, dtype/resource decisions, current record keys, and observation/identity fields. Do not insert a full `snapshot()` into a hot numerical/admission path.

### Recipe C: lexical contexts

Replace upstream setting contexts with a `NumericsConfig` context over the same extent and values. Preserve nesting and exception restoration. Include dtype/rank promotion, directional transfer guard, and disable-JIT contexts, not only x64. Do not add decorators to scientific callables.

### Recipe E: process environment

Use native runtime environment projection for JAX/Phydrax launch keys before child startup, while preserving the existing owner's base inheritance, setdefault-versus-override policy, XLA flag contents/order, CPU-only versus forced-device meaning and benchmark IPC. Shell deployment recipes may still spell authoritative external variables at the interchange boundary; Python examples should use the native projection.

### Required per-file exceptions

| File(s) | Non-mechanical obligation |
| --- | --- |
| `tools/differential_operator_benchmarks.py` | Retain `arguments.dtype == "float64"`; x64 is a trace/process setting, not a backend topology setting. Preserve both dtype campaigns and existing per-record cache policy |
| `tools/signature_benchmarks.py` | Resolve the requested platform before backend work; preserve its conditional `--platform` contract and refuse genuinely late changes |
| `tools/nuclear_fuel_qualification.py`; `tests/unit/applications/nuclear_fuel/test_qualification.py`, `test_report_aggregation.py`, `test_rz_manufactured.py`; `benchmarks/nuclear_fuel_performance.py`; `examples/nuclear_fuel_base_irradiation.py`, `nuclear_fuel_calibration.py`; `docs/guides_nuclear_fuel.md` | Preserve the imported qualification helper's float64 requirement at every consumer; do not change global-setting lifetime accidentally |
| `tools/distributed_substrate_benchmark.py` | Replace removed `initialize_from_environment` import/call with config startup plus execution `initialize`; retain local launch and process/runtime facts |
| `tools/numerical_interoperability_qualification.py`; `benchmarks/numerical_interoperability.py` | Retain forced-host child topology and fresh-interpreter row discovery. Do not remove isolation just because a module-scope setter has moved |
| `tests/unit/discretization/test_unstructured_finite_volume.py` | Migrate the embedded false-x64 child and query; preserve large-ID lossless/refusal behavior and child isolation |
| `tests/interchange/test_mace_checkpoint.py` | Migrate the embedded true-x64 native-without-provider child; retain real model evaluation/restore and external-provider absence |
| `examples/energy/_artifacts.py`; `tools/battery_qualification.py` | Preserve the same effective observation fields in environment/runtime identity; do not replace them with a new configuration ID or change scientific numeric IDs |
| `tests/unit/discretization/test_distributed_spectral.py` | Keep its precision-dependent numerical-ID contract; migrate only the configuration observation |
| `tests/integration/test_docs_examples.py` | Keep exact executable fence parsing, per-document block composition, child environment and timeout policy |

## 7. Tests: additions and existing contracts

New permanent tests must own consumer-visible boundaries, not implementation wiring, forwarded arguments, source text, copied defaults, or generic not-throw assertions. Fresh-process scenarios are necessary for startup/backend contracts; ordinary numerical tests retain natural topology.

| File | Planned contracts |
| --- | --- |
| **New `tests/unit/test_configuration_resolution.py`** | Required fields; independent-layer precedence; nested resolution; unknown/derived paths; duplicate/ancestor conflicts; scalar/union ambiguity; finite values; bounded literal grammar; executable expression/method refusal; atomic failure; independent mutable/default input isolation; exact typed result |
| **New `tests/unit/test_configuration_runtime.py`** | Fresh-process default/explicit x64 behavior; effective getters inside native contexts; supported compound-context restoration on errors; no cross-thread propagation claims; backend-bound late-change refusal without false mutation; unchanged/idempotent requests; pre-import-only refusal |
| **New `tests/unit/test_configuration_environment.py`** | Deterministic explicit mappings, native layer precedence, no input/process mutation, absent-versus-empty host settings, launcher joint-field refusal, ordered rank/local-device values, child projection preserving unrelated variables and flag text |
| **New `tests/unit/test_configuration_records.py`** | Canonical order/content identity independent of origin layer, lossless supported values, exact-field/ID tamper refusal, no arrays/live resources or credential/ambient-environment inclusion |
| **New `tests/integration/test_configuration_startup.py`** | Actual cold root/config import leaves backend uninitialized; direct-submodule and lazy-root numerical access both work with prior settings; local discovery occurs at execution initialization; two-rank coordination happens before device use; existing JAX-first import and late bootstrap refusal |
| `tests/conftest.py` | Route `strict_jax` dtype/rank contexts and logging fixture setup/teardown through config. Keep marker behavior and logging sink lifecycle; no global device-count changes |
| `tests/unit/test_logging.py` | Migrate setup/read APIs; retain default silence, observable event content, JSON/text sinks, handler independence, redaction, correlation scope and restoration |
| `tests/unit/test_distributed_execution_substrate.py` | Native collectives observation, config-owned bootstrap type, rank overlay, local runtime facts, pre-device coordination and late initialization refusal |
| `tests/unit/test_plugin_registry.py` | Retain exact registry identity, duplicate/conflict refusal and custom registration; add real cold built-in restoration at its consumer, not a registry import-spelling test |
| `tests/unit/nn/test_operator_loader.py`; `tests/unit/atomistic/test_model_artifact.py`; `tests/interchange/test_mace_checkpoint.py` | Real portable cold restore, exact field/array inventories, consumer evaluation, unsupported/unknown ID refusal and no external provider requirement for native restore |
| `tests/unit/interchange/test_cosmology_formats.py` | Real public cosmology interchange round-trip after lazy access; do not test private lazy-map wiring |
| `tests/unit/geometry/test_meshcore_loader.py`; `tests/unit/meshing/test_distribution.py` | Existing lazy configured-path/fallback/ABI/index-width contracts; add absent/blank versus invalid-configured-path edges only where not covered |
| `tests/unit/meshing/test_provider_worker.py` | Explicit/env/default precedence, present-empty behavior, delayed resolution, child environment versus parent PATH separation, actual session behavior and relaunch |
| `tests/unit/meshing/test_mmg_provider.py`, `test_omega_h_provider.py`, `test_tioga_provider.py`, `test_vorocrust_provider.py` | Use native worker-setting resolution in real-provider setup; retain canonical missing-provider skips and live behavior assertions |
| `tests/interchange/test_energy_runtime.py` | Preserve inheritance/allowlist/override/temp-directory/memory-limit behavior and privacy; no generic second environment builder |
| `tests/unit/test_commercial_service.py` | Preserve instance isolation, K8s field references/rank environment, request digest/idempotency and resolved-run persistence |
| `tests/unit/lifecycle/test_resolved_run.py`; `tests/unit/lifecycle/test_migration_registry.py`; `tests/unit/solver/test_production_runtime.py` | Bind a native resolved record through the existing prepared-configuration reference; retain canonical run IDs, forward-only migration, explicit lossy/ambiguous refusal, lineage, rollback and restart admission |
| `tests/unit/interpolation/test_rectilinear.py`; `tests/unit/test_packed_bvh.py`; `tests/unit/applications/test_neural_events.py`; `tests/unit/applications/test_geophysical_contracts.py` | Preserve consumer-visible dtype/index capacity and x64-dependent refusal after read migration |
| `tests/unit/applications/test_almonacid_2024_continuum.py`; `tests/unit/optim/test_cones.py`, `test_cone_barriers.py`; `tests/unit/applications/test_robot_environment.py` | Preserve scientific precision gate, numerical cone behavior and effective PRNG provenance |
| `tests/unit/test_linalg_functional_precision.py`; `tests/unit/nn/test_operator_precision.py`; `tests/unit/solver/test_projector_monte_carlo_precision.py` | Preserve scoped policy precedence, inheritance versus explicit default, dtype/promotion, differentiation and failure semantics |
| `tests/unit/typing/test_convert.py`; `tests/unit/typing/test_strict_integration.py` | Preserve actual transfer-guard behavior, conversion exactly once and reconstruction validation |
| `tests/unit/test_constraint_public_api.py`, `test_high_dimensional_public_api.py`, `test_stage10_public_api.py` | Exercise existing consumer surfaces after lazy-root change; do not re-pin incidental import-order assertions |
| `tests/integration/test_import_boundaries.py` | Extend the existing audit's own diagnostic contract with alias/subprocess/fence examples; no source-text product tests |
| **New `tests/typing/cases/configuration.py`** | Pin generic resolved result types, native request/context/getter signatures and expected static misuse with exact line-local ty diagnostics |
| **New `tests/typing/installed/configuration.py`** | Pin installed-wheel config and lazy-root exports outside the repository, with no `Any` leakage |

The existing source-derived tests in appendix A must be migrated without deleting their scientific assertions or broadening skips. If a touched test asserts only incidental wording/wiring, remove that incidental test rather than re-pin it. Do not introduce a suite-architecture refactor or broad new property matrix.

### Edge-case matrix

1. Environment unspecified, true, false, empty, malformed, and partially specified bootstrap tuples.
2. JAX imported before Phydrax, a backend initialized before a changed request, identical late requests, and external contexts masking process values.
3. Nested native contexts with different fields, a failed context entry, an exception in execution, and restoration after all of them.
4. No override versus explicit false versus explicit matmul default; preserve existing native policy subsets.
5. Integer IDs at int32 boundaries and beyond; x64-off refusals for required int64/uint64, without narrowing/clipping.
6. Required nested input, constructor-derived field, unsupported class/form, bool-as-int, lossy numeric conversion, union ambiguity, nonfinite data, cyclic/oversized input and duplicate/overlapping override paths.
7. Explicit provider path, inherited setting, unset, blank and present-empty worker values; configured-invalid path must not fall back.
8. Parent PATH selection versus child allowlisted environment; policy memory-limit value wins over caller overrides.
9. Cold built-in artifact restore, custom registered values, unknown/spoofed IDs and optional-provider absence.
10. Equal canonical configuration values from different input origins; meaningful ordered tuples; record tampering; existing run-spec reference without numerical-revision changes.
11. False-x64 worker projection, supported float32 execution or honest scientific refusal, runtime mismatch and failed publication.
12. Logging already enabled before importing `phydrax.logging`; unrelated application handlers; nested correlation context and sink teardown.

## 8. Architecture enforcement, docs, and generated data

| File | Planned action |
| --- | --- |
| `pyproject.toml` | Add Ruff banned upstream config APIs with an exact owner exception for `_jax_config.py`. Retain current Python dependency range, warning policy, dtype/rank marker semantics and pinned tools |
| `tools/check_import_boundaries.py` | Extend the existing boundary owner to detect upstream config access/aliases, config contexts, first-party native JAX launch-key construction and known embedded Python scripts. Check maintained Python documentation fences. Keep historical source quotations explicitly classified, not blanket-exempt all docs |
| `tools/generate_public_api_manifest.py` | Include the config facade through root exports; retain lazy discovery, explicit cosmology module and deterministic ordering. Do not require eager resolution of optional leaf providers |
| `tools/check_public_api_manifest.py` | Verify the canonical cutover after regeneration; no alternate public configuration paths |
| `tools/generate_capability_inventory.py`; `tools/check_capability_consistency.py` | Regenerate/check current inventory after public API change. Do not invent scientific release profiles or promote a capability merely to advertise a configuration API; unchanged catalog outputs may remain unchanged |
| `mkdocstrings_setup.py` | Preserve static manifest-backed facade binding; adjust only if the new explicit exports require it, not by importing the package during documentation analysis |
| **New `docs/api/configuration.md`** | Document supported records/operations, configuration versus runtime facts, effective reads, native contexts, exact phases/errors, records, and the one-adapter contract |
| **New `docs/guides_configuration.md`** | Python-first immutable recipes, dotted literal overrides, required inputs, declared origins, explicit configure/context/child projection, show-only resolution without execution, and scientific policy composition |
| `docs/api/execution.md` | Move bootstrap-input documentation to config; remove old initialization helper; retain execution initialization/discovery/topology and rank launch semantics |
| `docs/api/logging.md`; `docs/guides_logging.md`; `docs/guides_solver.md` | Move emission configuration calls, including the solver guide's logging example; retain sink operations/correlation/privacy/default-silent behavior and explicit telemetry |
| `docs/api/phydrax.md`; `README.md` | Add canonical configuration navigation/overview and document lazy-root/default changes, without introducing root aliases |
| `docs/guides_testing.md` | Document configuration contexts, startup subprocesses and justified global verification; no global forced topology |
| `docs/guides_meshing.md`; `docs/api/geometry.md`; `docs/examples/index.md` | Document native host inputs and exact worker/library selection/failure rules |
| `docs/api/export.md` | Document native worker projection/replay, retained isolated environment and explicit runtime/scientific refusal |
| `docs/guides_solver_evidence.md`; `docs/api/solver/functional_solver.md` | Describe config-record reference and native context application without moving precision-policy/evidence ownership |
| `docs/guides_particle_in_cell.md`, `docs/electromagnetic_radiation_sources.md`, `docs/guides_charged_particle_radiation.md` | Keep external pinned-provider deployment contracts distinct from settings actually read by Phydrax |
| `mkdocs.yml` | Add guide/API navigation using canonical paths |
| `docs/data/public_api.json` | Regenerate after the final implementation delta; remove old exports and add config |
| `docs/data/capabilities.json`, `docs/api/capabilities.md`, other outputs of the current capability generator | Regenerate deterministically if affected; do not hand-edit generated records |
| `CHANGELOG.md` | Record default/environment behavior change, device-inert imports, configuration API, removed bootstrap/logging paths, native JAX boundary and identity implications |
| `NOTICE`, `LICENSES/*`, dependency/lock files | No external code or dependency is adopted, so no attribution/dependency changes are planned |

All maintained direct configuration snippets and launch recipes in appendix A receive their recipe-specific migration. Keep `python executable` markers exactly where an example is intended to run: the current docs runner recognizes that exact opener, combines a document's blocks, and runs one child per document. Ordinary Python fences are not evidence of exercised examples.

## 9. Identity, compiler, performance, and verification policy

### 9.1 Identity limits

This is a clean configuration/API cutover, not a guarantee that all historical source-derived IDs stay equal. `_identity.py` hashes function code and imported bindings; changing a config-reading function or context helper can change source/compiler identity even when numerical results are equal. Moving `RuntimeBootstrap` changes its type ownership.

Preserve existing scientific fields, PyTree order/partition, RNG addressing, archive payloads, policy IDs, provider IDs and numeric inputs. Do not modify `_identity.py` to fake old payloads, unwrap arbitrary helper wrappers, relax opacity, or ignore new imported helper state.

Native context/read helpers used from compiled scientific functions must be stateless where required by compiler admission. Do not capture array-bearing owners, add wrappers/closures to callbacks, or expose state locks/schema plans as compiler metadata. Test actual JVP/VJP/compiler-bound workflows before completing the broad callsite migration.

No existing static/dynamic partition change is planned. If implementation discovers that one is necessary, obtain explicit approval for that representation change before doing it; do not call it a typing/configuration cleanup.

### 9.2 Verification selection and order

Use one integration-owned evidence record. Record exact command/scenario, relevant source state, toolchain/backend/device/environment identity, pass/fail/skip/block, and exact failing node IDs. Do not repeatedly run valid evidence on unchanged inputs.

Focused wave, after each complete integrated slice and formatting:

1. Run the actual changed surface: cold import/configure, resolve/show-only, a real numerical computation in a native context, provider selection, or a child launch.
2. Run direct failing/focused consumer nodes with `-n auto`, except dedicated provider/topology processes.
3. Run changed-file Ruff and ty diagnostics, selector/contract/declaration checks and the config boundary audit.
4. Fix narrow failures before expanding coverage.

Integration wave:

- `python tools/check_typing.py check`
- `python tools/audit_selectors.py`
- `python tools/audit_contract_candidates.py --signatures`
- `python tools/check_import_boundaries.py`
- `python tools/check_public_api_manifest.py`
- `python tools/check_capability_consistency.py`
- `python -m tools.check_installed_typing` for the actual supported Python default (currently 3.12). Do not widen package support to 3.11/3.13 as part of this change.
- Run the affected matrix in section 7/appendix A, plus persistence/compiler consumers reached through changed symbols.
- Do not run the full suite again. The user's strictly-affected-tests restriction and the matrix in [native-configuration-remediation.md](native-configuration-remediation.md) control further work, including shared startup/typing changes.

Specialized wave, separate from unrestricted ordinary tests:

- Forced CPU topology children for restart migration, distributed meshing and PIC.
- A real two-process local coordination smoke with process-index/count and numerical collective assertions.
- Real optional providers only at their owning boundaries; installed providers are not silently skipped.
- Native builds only if actual native files/contracts change. None are required merely because host setting lookup moved.
- Accelerator/multi-host claims require actual corresponding topology; forced CPU devices establish functional contracts only.

After final source integration, regenerate public API/capability artifacts once, then rerun only gates invalidated by that delta plus the required final matrix. Do not modify old measured JSON as a surrogate for rerunning a benchmark.

### 9.3 Minimal smoke and regression commands

The commands below record the original migration selection, not a mandate to rerun it. Further remediation uses only the strictly affected files/nodes in [native-configuration-remediation.md](native-configuration-remediation.md); no broad integration/docs-example selection or full suite is authorized.

```sh
python -m pytest -q -n auto tests/unit/test_configuration_resolution.py tests/unit/test_configuration_runtime.py tests/unit/test_configuration_environment.py tests/unit/test_configuration_records.py tests/integration/test_configuration_startup.py
python -m pytest -q -n auto tests/unit/test_logging.py tests/unit/test_distributed_execution_substrate.py tests/unit/typing/test_convert.py tests/unit/typing/test_strict_integration.py tests/unit/test_linalg_functional_precision.py tests/unit/nn/test_operator_precision.py tests/unit/solver/test_projector_monte_carlo_precision.py
python -m pytest -q -n auto tests/unit/nn/test_operator_loader.py tests/unit/atomistic/test_model_artifact.py tests/interchange/test_mace_checkpoint.py tests/unit/test_plugin_registry.py
python -m pytest -q -n auto tests/unit/lifecycle/test_resolved_run.py tests/unit/lifecycle/test_migration_registry.py tests/unit/solver/test_production_runtime.py
python -m pytest -q -n auto tests/typing/test_typecheck.py tests/integration/test_import_boundaries.py tests/integration/test_docs_examples.py
```

Run these in dependency-aware waves, not concurrently as several `-n auto` invocations. During iteration use exact failing nodes, not the entire command list again. Use the already-owned topology subprocess tests rather than altering root test configuration.

Actual runtime smoke requirements, beyond tests:

- A fresh interpreter imports root/config, resolves and prints a record, observes backend-not-ready, then applies a request and performs a real supported dtype/promotion calculation.
- Separate fresh interpreters execute direct geometry import and lazy-root geometry access; verify a public numerical result, not an internal quadrature constant.
- A child environment with explicit x64 false proves the native effective query and supported float32 result, or the owning scientific refusal for an int64/float64-only route.
- Local execution initialization produces actual process/platform/device facts only at that action.
- A real two-rank local launch performs a collective numerical calculation with correct process indices/count.
- A cold portable native model/operator artifact is restored and evaluated without root preloading or its external provider.
- Existing `examples/finite_element_poisson.py` and a representative changed configuration-bearing example run on their actual surfaces.

### 9.4 Benchmarks

Add **`benchmarks/configuration.py`**, using current benchmark runtime/evidence helpers, for the following separately reported phases:

- Fresh-process cold imports: JAX control, root, root/config access, direct geometry access.
- Peak host RSS and backend readiness for those cases; imported-module observations are benchmark diagnostics, not permanent wiring tests.
- Host configuration resolution/application and bounded override-count scaling, executed outside numerical loops.
- Representative numerical-context lowering, compilation, warm execution and compiler temporary/output/code bytes, plus logical retained bytes where material.

Capture startup/runtime baselines before core edits. New resolver functionality has no historical baseline; report that fact instead of fabricating one. Use the same material environment for comparable cases, follow cache policy and run without tests/builds/agents contending for resources. No repeated wrapper construction, snapshot allocation or host synchronization may enter warmed numerical iterations.

Run relevant existing phase-separated benchmarks/qualification smoke for changed precision/context or launcher owners, including the existing distributed substrate benchmark and conditional dtype differential-operator benchmark. Record supported outcomes; do not infer speedup or device scalability from a smaller import time.

Measure materially changed helper complexity before/after with the repository-required signals. Ordinary new helpers target cyclomatic complexity at most 15, cognitive complexity at most 20, and at most 120 NLOC. Decompose by resolution, validation, phase admission, application/restoration and record construction; no trivial forwarding helpers to game metrics.

## 10. Dependency-aware implementation ownership

1. **Integration/core owner:** freeze the API/record/phase contract; implement the new configuration owner, root startup, bootstrap split, facade cutover and adapter. Prove cold import and one actual context/compiler path before releasing broad migrations.
2. **Scientific caller owner:** own all package rows in section 4, including the logging imports in functional gradient/evolution/KFAC. Preserve source/field identities deliberately and report exact scientific/compiler consumers.
3. **Library/provider owner:** own section 5 loader/worker/scheduler changes and their tests, consuming the fixed core environment interface. Do not edit core state/environment files concurrently.
4. **Frontend owner:** own appendix A Python callers and embedded scripts, preserving process/context timing and subprocess isolation. Own `tests/conftest.py` and existing logging API callsite updates; coordinate its contract with the core owner.
5. **Contract-test owner:** own new config tests/typing fixtures and cold restore/record integration additions, not frontend fixtures or production files.
6. **Documentation/integration owner:** update authored docs/changelog, boundary tooling and manifests after implementation interfaces stabilize; integrate all slices and run the evidence waves once.

Core startup is a true prerequisite. Once it passes, scientific callers, providers, frontend callers and new contract tests are independent slices with explicit file ownership. No two agents edit the same shared helper/fixture. Agents skip builds/lint/tests/formatters mid-flight; the integration owner schedules verification after each complete integrated wave. Allocate one total CPU/device/native-build resource budget rather than several unrestricted campaigns.

## 11. Completion criteria

- Every current first-party direct JAX setting read/write/context and native JAX launch-key construction is either migrated or has an exact legitimate external/worker/historical boundary disposition; the audit reports no unauthorized bypass.
- Root/config import is device-inert, both numerical import routes retain startup ordering, and real distributed coordination precedes device work.
- Explicit false-x64 startup is respected; required-width/precision scientific refusals and existing scoped precision policy semantics remain.
- Typed resolution, bounded literal overrides, required values, origins, canonical records and strict decoding work end to end without executable configuration or mutable wrappers.
- One config-owned logging flag exists; no old logging/bootstrap configuration exports, aliases or helper paths remain.
- Lazy library/worker selection, child security/environment precedence, provider identity, instance-local service configuration and lifecycle lineage remain intact.
- Cold artifact reconstruction works without root preload; no registry or identity workaround has been introduced.
- No new numeric arrays, keys, callback captures, scopes, locks or checker/schema state are hidden in static/device execution data.
- Public callers, annotations, typing fixtures, executable docs, examples, generated data and changelog use the same vocabulary.
- Smoke, focused contracts, required typing/audits, justified global suite and separate specialized evidence are recorded honestly; unresolved real prerequisites are stated rather than replaced by mocks/fallbacks.

## 12. Implementation outcome and deviations

The plan was implemented on branch `native-configuration`. Deviations and
additions discovered during implementation:

- `execution.initialize` was removed with `initialize_from_environment`; it
  duplicated `config.configure(config.RuntimeConfig(bootstrap=...))` followed by
  `execution.runtime_info()`. CPU collectives are selected only by
  `RuntimeConfig`.
- `LaunchEnvironmentConfig` carries only `xla_flags` and `preallocate`, the
  launch settings existing launchers use. Mapping fields are replaced whole
  during resolution. Record persistence lives in `_config_resolution.py`; no
  separate records module exists.
- Forced CPU topology uses `RuntimeConfig(cpu_device_count=N)` and
  `JAX_NUM_CPU_DEVICES`; `snapshot()` also observes the XLA flag form.
- Live-device placement scopes (`jax.default_device`) route through
  `_jax_config.default_device`; a device is a runtime object, not a record.
- Lazy root: the 64-bit startup choice is applied before any other Phydrax
  module loads, because `phydrax.typing` fixes canonical dtype rules at import.
  Every deferred load (`phydrax.<name>`, facade re-exports, cold artifact
  restore) runs under `_jax_config.process_numerics()`, so a caller's scoped
  `context` never fixes a module's import-time constants.
- Direct import of every public subpackage required import-cycle hardening:
  leaf modules import from leaf owners instead of hub facades, and 36 hub facades
  defer their re-exports through `phydrax._deferred_exports` with
  `TYPE_CHECKING` imports for static typing. No class moved, so type identities
  are unchanged. Cold artifact restore loads built-in registrations from the
  fixed public module list on a registry miss. Deferred re-exports are absent
  from a facade's `vars()` until first use, so namespace introspection uses
  attribute access or `dir()` (the native ML artifact registry, advanced-solver
  benchmark availability, and export-surface tests).
- `ProviderWorker` takes `setting: WorkerSetting` rather than a raw environment
  key.
- Unrelated failures found by verification were fixed: one-step
  `checkpointed_scan` runs and one-step replay blocks apply their body without a
  loop, which lets the differentiated single-tile periodic MACE export pass the
  IREE alias audit (multi-step differentiated scans remain refused under
  iree-base-compiler 3.11, because JAX's own residual tapes start from merged
  constants), and block-AMR cut-cell tests skip at the optional meshcore
  provider boundary.

## 13. Review remediation

The review identified 13 correctness/lifecycle/persistence defects plus an
embedded-script alias enforcement gap. The file-by-file corrective plan is
[native-configuration-remediation.md](native-configuration-remediation.md).
It includes an intentional canonical union/enum encoding and ID cutover,
plural environment provenance, dynamic JAX versus exact NumPy dtype contracts,
native environment decoding, faithful bootstrap projection, partial-state
evidence, and complete lazy attribute scope/device protection.

The fixes are implemented; the remediation plan's status section records the
outcome and its deviations. No further full-suite run is permitted; the
corrective plan's strictly affected verification is authoritative.


## Appendix A. Exhaustive non-package source migration inventory

These are the live source paths found during planning. Each directory prefix applies to every filename listed beneath it. Apply the named recipe in section 6; several files deliberately appear in more than one group because they own more than one setting boundary. Historical/generated records are separately classified in A.8.

### A.1 Process x64 writes: tests (recipe P)

- `tests/unit/applications/`: `test_advanced_black_hole_perturbations.py`, `test_atomic_quantum_closure.py`, `test_biomembrane.py`, `test_black_hole_perturbations.py`, `test_black_hole_self_force.py`, `test_black_hole_thermodynamics.py`, `test_cardiovascular_devices.py`, `test_cardiovascular_ep_reaction.py`, `test_cosmology_dark_matter_inference.py`, `test_cosmology_dark_matter_observables.py`, `test_distributed_qcd_production_closure.py`, `test_electrical_machine_design.py`, `test_electrophysiology.py`, `test_extended_black_hole_thermodynamics.py`, `test_frontier_conformal_bootstrap.py`, `test_frontier_curved_spacetime_qft.py`, `test_frontier_supersymmetric_lattice.py`, `test_gr_rays.py`, `test_gr_transfer.py`, `test_neural_events.py`, `test_neural_network.py`, `test_phase_field_defect_memory.py`, `test_phase_field_extended_closure.py`, `test_phase_field_multiphysics_closure.py`, `test_phase_field_workflows.py`, `test_point_neurons.py`, `test_qft_cavity_quantum_closure.py`, `test_qft_semiclassical_qed_closure.py`, `test_semiconductor_advanced_runtime.py`, `test_semiconductor_analysis.py`, `test_semiconductor_interfaces.py`, `test_semiconductor_quantum_advanced.py`, `test_semiconductor_traveling_wave.py`, `test_semiconductor_workflows.py`, `test_sonata.py`, `test_twisted_n2_rhmc.py`.
- `tests/unit/applications/nucleic_acid_biophysics/test_electronic_execution.py`.
- `tests/unit/discretization/`: `test_chebyshev_differentiation_action.py`, `test_discrete_field_views.py`, `test_finite_element_field_view.py`, `test_learned_energy_equilibrium.py`, `test_learned_thermal_research.py`, `test_prepared_field_queries.py`, `test_side_actions.py`, `test_smooth_compressible_d2v_forcing.py`, `test_spectral_field_view.py`.
- `tests/unit/electromagnetics/`: `test_gyrosynchrotron_providers.py`, `test_magnetobremsstrahlung.py`, `test_thermal_synchrotron.py`.
- `tests/unit/export/test_discrete_velocity_iree.py`.
- `tests/unit/finance/core/`: `test_identifiers_currency.py`, `test_laws_evidence.py`, `test_time.py`.
- `tests/unit/finance/exposure/test_collateral_xva.py`; `tests/unit/finance/models/test_rates_credit.py`; `tests/unit/finance/portfolio/test_portfolio_compilation_replay.py`; `tests/unit/finance/risk/test_risk_scenarios_backtests.py`.
- `tests/unit/test_linalg_sparse_lu_analysis.py`; `tests/unit/test_sparse_derivatives.py`.
- `tests/unit/linalg/`: `test_hermitian_spectral_derivatives.py`, `test_pfaffian.py`, `test_sparse_ordering.py`.
- `tests/unit/metrix/`: `test_black_hole_metrics.py`, `test_gaussian_entanglement_closure.py`.
- `tests/unit/nonlinear/`: `test_implicit_fixed_point.py`, `test_lagged_linear_update.py`.
- `tests/unit/optics/wave/`: `test_coupled_mode.py`, `test_cylindrical_unidirectional.py`, `test_material_response.py`, `test_nonlinear_response.py`, `test_unidirectional_propagation.py`.
- `tests/unit/sampling/test_production_rhmc_pseudofermion.py`; `tests/unit/solver/test_distributed_line.py`; `tests/unit/tensor_network/test_frontier_anyon_categories.py`.

Embedded child scripts, still recipe P but retaining process isolation:

- `tests/interchange/test_mace_checkpoint.py`: true x64.
- `tests/unit/discretization/test_unstructured_finite_volume.py`: false x64 and width/refusal query.

### A.2 Process x64 writes: tools and benchmarks (recipe P)

- `tools/`: `adaptive_cubature_benchmarks.py`, `advanced_optics_benchmarks.py`, `almonacid_2024_continuum_qualification.py`, `almonacid_2024_refinement_qualification.py`, `black_hole_qualification.py`, `building_energy_benchmarks.py`, `calabi_yau_qualification.py`, `cm_quantum_lattice_qualification.py`, `cm_quantum_symmetry_qualification.py`, `cm_semiconductor_detector_qualification.py`, `column_radiation_qualification.py`, `conformal_bootstrap_qualification.py`, `dae_benchmarks.py`, `differential_operator_benchmarks.py`, `fourier_modal_maxwell_benchmarks.py`, `fourier_modal_maxwell_qualification.py`, `geomechanics_qualification.py`, `high_dimensional_pde_benchmarks.py`, `iga_h1_qualification.py`, `incompressible_ou_qualification.py`, `interactive_moist_column_qualification.py`, `large_eddy_simulation_qualification.py`, `learned_physical_design_benchmarks.py`, `meshfree_qualification.py`, `moist_column_qualification.py`, `multipoint_design_benchmarks.py`, `nuclear_fuel_qualification.py`, `optics_benchmarks.py`, `optics_qualification.py`, `semiconductor_laser_benchmarks.py`, `semiconductor_laser_qualification.py`, `singular_subspace_benchmarks.py`, `skeletal_muscle_energetics_qualification.py`, `skeletal_muscle_structured_fiber_qualification.py`, `skeletal_muscle_thermal_qualification.py`, `supersymmetric_lattice_qualification.py`, `surface_exchange_qualification.py`, `system_optics_qualification.py`, `taylor_gradient_diagnostic.py`, `thin_film_optics_qualification.py`.
- `tools/uq_benchmarks/`: `gp_scaling.py`, `nuts_scheduling.py`.
- `benchmarks/`: `compatible_projection_scaling.py`, `effective_resistance_scaling.py`, `memory_astrodynamics_pairs.py`, `memory_field_view.py`, `memory_host_caches.py`, `memory_host_exact.py`, `memory_kernel_correction.py`, `memory_ocean_shell.py`, `memory_ringdown.py`, `meshfree_scaling.py`, `nuclear_fuel_performance.py`, `numerical_interoperability.py`, `rsp_free_energy_gate.py`, `shortest_path_field_scaling.py`.
- Additional platform writer: `tools/signature_benchmarks.py`, preserving its explicit conditional platform request.

### A.3 Process x64 writes: examples (recipe P)

`examples/`: `adaptive_fe_fv_rebind.py`, `advanced_bubble_cloud.py`, `almonacid_2024_continuum.py`, `binary_black_hole_extraction.py`, `cherenkov_prescribed_charge.py`, `compact_object_accretion.py`, `coupled_control.py`, `coupled_data_assimilation.py`, `coupled_inverse_problem.py`, `coupled_lane_worksets.py`, `coupled_learned_interface.py`, `coupled_rom_swap.py`, `coupled_scalar_regions.py`, `coupled_transient_fields.py`, `dispersive_pic_cherenkov.py`, `double_point_mpm.py`, `em_shower_slab.py`, `enforced_periodic_constraints.py`, `fixed_grid_z4c.py`, `fourier_modal_grating.py`, `fourier_modal_localized_emitter.py`, `geophysical_flux_closure.py`, `grrmhd_radiation_feedback.py`, `hybrid_pinn_classical.py`, `interactive_moist_column.py`, `kerr_horizon_thermodynamics.py`, `kerr_shadow_rays.py`, `magnetobremsstrahlung_emission.py`, `meshfree_fluid_structure.py`, `meshfree_mixed_method_coupling.py`, `meshfree_production_epochs.py`, `meshfree_production_restart.py`, `moist_column.py`, `named_block_dae.py`, `nitsche_transmission.py`, `nuclear_fuel_base_irradiation.py`, `nuclear_fuel_calibration.py`, `plasma_ray_transfer.py`, `polarized_fast_light_grrt.py`, `prepared_field_observations.py`, `qnm_scattering_hawking.py`, `sem_vem_bem_transmission.py`, `simulation_product_visibility.py`, `skeletal_muscle_thermal.py`, `sonata_native_roundtrip.py`, `taylor_derivative_contractions.py`.

Remove `from jax import config` aliases when present and import the canonical native facade. Keep JAX imports needed by numerical code.

### A.4 Effective reads (recipe R)

- Tests: `tests/unit/test_linalg_mixed_precision.py`; `tests/unit/discretization/test_distributed_spectral.py`; `tests/unit/discretization/test_unstructured_finite_volume.py`; `tests/unit/ml/linear/test_least_sparse.py`; `tests/unit/nn/test_operator_precision.py`.
- `tools/`: `black_hole_qualification.py`, `cm_periodic_electronic_qualification.py`, `cm_transport_qualification.py`, `dry_atmosphere_qualification.py`, `dynamics_benchmarks.py`, `electrical_machine_design_benchmarks.py`, `geomechanics_qualification.py`, `global_atmosphere_qualification.py`, `high_dimensional_pde_benchmarks.py`, `incompressible_ou_qualification.py`, `ml_interoperability_qualification.py`, `numerical_interoperability_qualification.py`, `phase_field_extended_qualification.py`, `phase_field_multiphysics_qualification.py`, `phase_field_production_qualification.py`, `soap_film_tunnel_qualification.py`, `soft_transport_benchmarks.py`, `spherical_vector_qualification.py`, `state_space_gp_benchmarks.py`, `thin_film_optics_qualification.py`, `trefftz_benchmarks.py`.
- `tools/uq_benchmarks/report.py`; `benchmarks/exterior_calculus.py`.
- `examples/`: `interactive_column_inference.py`, `interactive_moist_column.py`, `monolithic_contact.py`, `point_cloud_poisson.py`.
- `examples/energy/`: `_artifacts.py`, `building_dispatch.py`, `power_fault.py`.

### A.5 Lexical x64 contexts (recipe C)

- `tests/integration/`: `test_electrophysiology_workflow.py`, `test_energy_workflows.py`, `test_finance_workflows.py`, `test_meshfree_learned_flux_training.py`, `test_semiconductor_laser_workflow.py`.
- `tests/unit/applications/`: `test_almonacid_2024_continuum.py`, `test_almonacid_2024_material.py`, `test_almonacid_2024_preconditioning.py`, `test_column_inference.py`, `test_geophysical_observations.py`, `test_interactive_moist_column.py`, `test_semiconductor_continuum.py`, `test_skeletal_muscle_thermal.py`, `test_tree_cable.py`, `test_reduced_climate.py`.
- `tests/unit/applications/nuclear_fuel/`: `test_calibration.py`, `test_ensembles.py`, `test_execution_profile.py`, `test_observables.py`, `test_reduction_consistency.py`, `test_restart.py`, `test_sensitivity.py`, `test_shared_execution.py`, `test_specification.py`, `test_step_transaction.py`, `test_thermomechanics_verification.py`, `test_trial_admissibility.py`.
- `tests/unit/combinatorial/test_capacitated_auction.py`.
- `tests/unit/discretization/`: `test_finite_volume_embedded_stage.py`, `test_structured_interface_geometry.py`, `test_unstructured_embedded_boundary_vof.py`.
- `tests/unit/discretization/meshfree/`: `test_conservation_solve.py`, `test_constitutive_laws.py`, `test_corrections.py`.
- `tests/unit/discretization/spatial/test_distributed_relations.py`; `tests/unit/metrix/test_legendre_geometry.py`; `tests/unit/nonlinear/test_trial_validity.py`.
- `tests/unit/operators/`: `test_convolution_quadrature.py`, `test_periodic_maxwell_boundary3d.py`.
- `tests/unit/solver/`: `test_dynamic_vector_cq.py`, `test_fem_bem_vector.py`, `test_maxwell_fem_bem_matching.py`, `test_projector_monte_carlo.py`, `test_unstructured_embedded_runtime.py`.
- `tests/unit/test_linalg_runtime.py`; `tests/unit/uq/test_gaussian_chain.py`, `tests/unit/uq/test_physical_priors.py`.
- `tools/`: `almonacid_2024_material_qualification.py`, `geophysical_assimilation_qualification.py`, `learned_energy_boundary_qualification.py`, `learned_energy_equilibrium_qualification.py`, `learned_energy_export_qualification.py`, `learned_energy_forcing_qualification.py`, `learned_energy_hybrid_qualification.py`, `learned_energy_off_lattice_qualification.py`, `learned_energy_spatial_qualification.py`, `learned_thermal_kinetic_qualification.py`, `reduced_climate_qualification.py`.
- `examples/`: `geophysical_assimilation.py`, `meshfree_learned_edge_flux.py`, `meshfree_learned_metric_correction.py`, `reduced_climate_scenarios.py`.
- `docs/guides_skeletal_muscle_continuum.md`.

Retain explicit false/parameterized/both-value scopes in finite-volume embedded stages, distributed relations, linalg runtime, projector Monte Carlo, unstructured embedded runtime, Gaussian chains, structured interface geometry, and unstructured embedded-boundary VOF.

### A.6 Other numerical contexts (recipe C)

- Promotion: `tests/conftest.py`; `tests/unit/linalg/test_compound.py`; `tests/unit/linalg/test_linear_solve_derivative_contract.py`; `tests/unit/solver/test_projector_monte_carlo_precision.py`.
- Directional transfer guard: `tests/unit/typing/test_convert.py`; `tests/unit/typing/test_strict_integration.py`.
- Disable JIT: `tests/unit/uq/test_guided_particle.py`; `docs/cookbook/operator_learning.md`.

### A.7 Environment launch/capture and topology (recipe E)

Python launch/capture owners:

- `tests/integration/test_docs_examples.py`; `tests/integration/test_uq_operator_benchmark.py`; `tests/integration/test_meshfree_runtime_restart.py`; `tests/integration/_meshfree_runtime_migration.py`.
- `tests/unit/meshing/test_device_adaptation.py`; `tests/unit/solver/test_distributed_pic_capabilities.py`; `tests/unit/test_distributed_execution_substrate.py`; `tests/unit/test_commercial_service.py`.
- `tools/exterior_calculus_qualification.py`; `tools/numerical_interoperability_qualification.py`; `tools/distributed_substrate_benchmark.py`; `tools/battery_qualification.py`.
- `benchmarks/_runtime.py`; `benchmarks/numerical_interoperability.py`; `benchmarks/distributed_spectral.py`; `benchmarks/meshfree_closure_distributed.py`.

Retain and update dedicated-topology invocation documentation/skip contracts:

- `tests/unit/atomistic/test_distributed_mace.py`; `tests/unit/discretization/meshfree/test_distributed.py`; `tests/unit/discretization/spatial/test_distributed_relations.py`; `tests/unit/solver/test_distributed_pic.py`; `tests/unit/coupling/test_execution_contracts.py`.
- `benchmarks/distributed_atomistic_scaling.py`.
- `docs/guides_atomistic_distributed_execution.md`; `docs/guides_geophysics.md`; `docs/guides_numerical_interoperability.md`; `docs/guides_phase_field.md`; `docs/guides_phase_field_multiphysics.md`; `docs/guides_two_phase_hydrodynamics.md`.

The native boundary distinguishes CPU selection, forced-host topology, physical topology and CPU threading controls. Existing observation fields remain evidence, not new ambient defaults. Preserve old measured environment spellings at the historical/external boundary; new native projection uses the supported upstream contract without silently rewriting historical IDs.

### A.8 Maintained documentation and explicitly retained material

Maintained Python snippets:

- Recipe P: `docs/guides_atomistic_distributed_execution.md`, `docs/guides_nuclear_fuel.md`, `docs/guides_semiconductor.md`, `docs/guides_sonata.md`, `docs/guides_statistical_incompressible_flow.md`, `docs/api/uq/correlated_ratios.md`, `docs/cookbook/graph_physics.md`, `docs/cookbook/projector_monte_carlo.md`.
- Recipe C: `docs/guides_skeletal_muscle_continuum.md`, `docs/cookbook/operator_learning.md`.

Maintained x64 launch recipes and scientific requirements:

- `docs/biophysical_sources.md`; `docs/guides_column_inference.md`; `docs/guides_dry_atmosphere.md`; `docs/guides_energy_planning.md`; `docs/guides_energy_workflows.md`; `docs/guides_fem_applications.md`; `docs/guides_geomechanics_qualification.md`; `docs/guides_global_atmosphere.md`; `docs/guides_meshfree.md`; `docs/guides_monolithic_contact.md`; `docs/guides_phase_field.md`; `docs/guides_phase_field_multiphysics.md`; `docs/guides_semiconductor.md`; `docs/cookbook/nucleic_acid_biophysics.md`; `docs/examples/index.md`; `docs/guides_projector_monte_carlo.md`.

Retained material, not active migration targets:

- `docs/plans/enforced-periodic-constraints.md` and `docs/plans/projector-quantum-monte-carlo.md` contain historical source/planning quotations. Preserve their historical meaning and explicitly distinguish them from maintained executable configuration guidance.
- Committed `benchmarks/*.json`, baseline JSON and prior qualification reports are measured evidence. Do not text-replace configuration fields or pretend the new implementation produced them.
- `mkdocstrings_setup.py` and `native/meshcore/python` have no direct upstream configuration caller in this inventory; only actual new facade/static documentation requirements justify changes.
- Existing scientific precision/resource policy definitions, instance-local service inputs and lifecycle record shapes remain intentionally unchanged as specified above.

## Appendix B. Source evidence and design rationale

Key current source anchors:

- `phydrax/__init__.py:6-19`: unconditional x64 and environment bootstrap; `:25-255`: eager package tree and cosmology side effect.
- `phydrax/geometry/__init__.py:34-45` and `geometry/analytic/__init__.py:6-18`: eager analytic import; `_extended.py:83-88`: device-backed quadrature constants.
- `phydrax/_execution_bootstrap.py:168-186`: runtime discovery enumerates devices; `:197-217`: collectives setting and distributed initialization.
- Installed `jax/_src/config.py:270-275,309-320`: global update versus thread-local context/restoration.
- Installed `jax/_src/xla_bridge.py:760-763,1200-1217`: non-initializing private readiness and native CPU-device validator; `:1127-1132`: process count traverses devices.
- `phydrax/_identity.py:1022-1036,1092-1174`: source binding and semantic/numeric separation.
- `phydrax/lifecycle/_resolved_run.py:146-154`: existing prepared-configuration and policy references.
- `phydrax/_external_runtime.py:600-643`: explicit overrides participate in identity while an inherited environment is constructed separately. Do not claim native configuration ID alone makes that environment hermetic.
- `phydrax/logging.py:39-88,413-427`: enabled state versus context/transport and old import-time reset.
- `tests/integration/test_docs_examples.py:29-53,103-128`: exact executable fences and child runner.

Design scope: Python-first declarations, dotted data overrides, required inputs, and resolved display are in scope. Out of scope by design: `eval`-style expression parsing, CLI method invocation, mutable finalization or object traversal, constructor-based lossy coercion, wrapper typing spoofs, and pickle or unrestricted YAML persistence. No source copying or dependency adoption is planned.

All runtime consequences identified here remain implementation verification obligations. This plan supplies no fabricated pass results, performance numbers, provider availability, or scientific qualification claims.
